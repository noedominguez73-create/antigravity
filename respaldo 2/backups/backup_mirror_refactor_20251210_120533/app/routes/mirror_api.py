from flask import Blueprint, request, jsonify, current_app, render_template
from app import db
from app.models import MirrorItem, MirrorUsage
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Initialize Blueprint
mirror_api_bp = Blueprint('mirror_api', __name__, url_prefix='/api/mirror')

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# --- Helper Functions ---

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def configure_genai():
    """Configure Gemini API key from environment."""
    # Ensure env is loaded
    project_root = os.path.dirname(current_app.root_path)
    load_dotenv(os.path.join(project_root, '.env'))
    
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        current_app.logger.warning("No GOOGLE_API_KEY or GEMINI_API_KEY found.")
        return False
    
    genai.configure(api_key=api_key)
    return True

def get_best_model(capability='image'):
    """
    Dynamically select the best available model based on capability.
    capability: 'image' (generation) or 'vision' (analysis)
    """
    try:
        all_models = genai.list_models()
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if capability == 'image':
            # Priority: 3-Pro > 2.5-Flash > Experimental > Any Image
            img_models = [m for m in valid_models if 'image' in m.lower()]
            if not img_models: return None

            # 1. Gemini 3 Pro
            model = next((m for m in img_models if '3-pro' in m), None)
            if model: return model
            
            # 2. Gemini 2.5 Flash
            model = next((m for m in img_models if '2.5-flash' in m), None)
            if model: return model
            
            # 3. Fallback to any experimental/other image model
            return img_models[0]
            
        elif capability == 'vision':
            # Priority: 1.5 Pro > 1.5 Flash > Pro Vision
            pro = next((m for m in valid_models if 'gemini-1.5-pro' in m), None)
            if pro: return pro
            
            flash = next((m for m in valid_models if 'gemini-1.5-flash' in m), None)
            if flash: return flash
            
            return valid_models[0] if valid_models else None
            
    except Exception as e:
        current_app.logger.error(f"Error listing models: {e}")
        # Robust Fallback Defaults
        if capability == 'image': return 'models/gemini-2.0-flash-exp-image-generation'
        return 'models/gemini-1.5-flash'

    return None

def build_inpainting_prompt(hairstyle, color, instructions):
    """Constructs a safe, structured prompt for image generation."""
    target_parts = []
    if hairstyle: target_parts.append(hairstyle)
    if color: target_parts.append(color)
    if instructions: target_parts.append(instructions)
    
    target_desc = ". ".join(target_parts) if target_parts else "Enhance hairstyle professional look"

    return (
        "Generate a photorealistic image. Apply a professional hair transformation (Inpainting) to the subject's hair.\n\n"
        "TARGET LOOK:\n"
        f"{target_desc}\n\n"
        "STRICT INSTRUCTIONS:\n"
        "Preserve the subject's face, skin texture, expression, and identity with high fidelity. Only modify the hair."
    )

def save_uploaded_file(file, subdir='mirror'):
    """Saves an uploaded file with a unique timestamped name."""
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    unique_filename = f"gen_{timestamp}_{filename}"
    
    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', subdir)
    os.makedirs(upload_folder, exist_ok=True)
    
    save_path = os.path.join(upload_folder, unique_filename)
    file.save(save_path)
    
    return unique_filename, save_path, f"/static/uploads/{subdir}/{unique_filename}"

# --- Routes ---

@mirror_api_bp.route('/admin')
def admin_page():
    return render_template('admin_mirror.html')

@mirror_api_bp.route('/items', methods=['GET'])
def get_items():
    category = request.args.get('category')
    query = MirrorItem.query.filter_by(is_active=True)
    if category:
        query = query.filter_by(category=category)
    
    items = query.order_by(MirrorItem.order_index, MirrorItem.created_at.desc()).all()
    return jsonify([item.to_dict() for item in items])

@mirror_api_bp.route('/items', methods=['POST'])
def upload_item():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    name = request.form.get('name')
    category = request.form.get('category')
    order = request.form.get('order', type=int, default=0)
    
    if not name or not category:
        return jsonify({'error': 'Missing name or category'}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # Save File
        filename, file_path, image_url = save_uploaded_file(file, subdir='mirror')
        
        # Analyze Image for Prompt (if configured)
        generated_prompt = "Manual override prompt required"
        if configure_genai():
            try:
                model_name = get_best_model(capability='vision')
                if model_name:
                    model = genai.GenerativeModel(model_name)
                    img = Image.open(file_path)
                    
                    if category == 'color':
                        prompt = "Analyze hair color professionally using terms like Balayage, Babylights, Ombre. Describe ONLY the color."
                    else:
                        prompt = "Describe the hairstyle (cut, texture, length) in detail for an AI generator. Be concise."
                        
                    response = model.generate_content([prompt, img])
                    generated_prompt = response.text.strip()
            except Exception as e:
                current_app.logger.error(f"Prompt generation failed: {e}")
                generated_prompt = f"Error: {str(e)}"

        # Save DB Record
        new_item = MirrorItem(
            name=name, 
            category=category, 
            image_url=image_url,
            prompt=generated_prompt,
            order_index=order
        )
        if category == 'color':
            new_item.color_code = request.form.get('color_code')
            
        db.session.add(new_item)
        db.session.commit()
        
        return jsonify(new_item.to_dict()), 201

    except Exception as e:
        current_app.logger.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)}), 500

@mirror_api_bp.route('/items/<int:id>', methods=['PUT'])
def update_item(id):
    item = MirrorItem.query.get_or_404(id)
    data = request.json
    if 'order_index' in data: item.order_index = int(data['order_index'])
    if 'name' in data: item.name = data['name']
    db.session.commit()
    return jsonify(item.to_dict())

@mirror_api_bp.route('/items/<int:id>', methods=['DELETE'])
def delete_item(id):
    item = MirrorItem.query.get_or_404(id)
    item.is_active = False
    db.session.commit()
    return jsonify({'message': 'Item deleted'})

@mirror_api_bp.route('/generate', methods=['POST'])
def generate_look():
    """Core endpoint for Hairstyle/Color generation."""
    try:
        # 1. Validation & Setup
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        hairstyle = request.form.get('hairstyle')
        color = request.form.get('color')
        instructions = request.form.get('instructions')
        
        # 2. Save Input Image
        filename, save_path, result_image_url = save_uploaded_file(file, subdir='mirror')
        
        # 3. Configure AI
        if not configure_genai():
            return jsonify({'error': 'AI Configuration Missing'}), 500
            
        # 4. Construct Prompt
        full_prompt = build_inpainting_prompt(hairstyle, color, instructions)
        current_app.logger.info(f"Generating with prompt: {full_prompt}")

        # 5. Select Model
        model_name = get_best_model(capability='image')
        ai_description = "Processing..."
        
        if not model_name:
            # Fallback to text model if no image gen
            model_name = get_best_model(capability='vision')
            current_app.logger.warning("No Image Gen model found. Falling back to text.")

        model = genai.GenerativeModel(model_name)
        img = Image.open(save_path)
        
        # 6. Generate
        try:
            # Relax Safety for "Face Editing"
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            is_image_model = 'image' in model_name.lower() or 'exp' in model_name.lower()
            
            if is_image_model:
                current_app.logger.info(f"Calling Image Model: {model_name}")
                response = model.generate_content([full_prompt, img], safety_settings=safety_settings)
                
                # Extract Image
                image_found = False
                if response.candidates and response.parts:
                    for part in response.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # Save Generated Image
                            gen_filename = f"gen_out_{os.path.basename(filename)}.png" 
                            gen_save_path = os.path.join(os.path.dirname(save_path), gen_filename)
                            
                            with open(gen_save_path, 'wb') as f:
                                f.write(part.inline_data.data)
                                
                            result_image_url = f"/static/uploads/mirror/{gen_filename}"
                            ai_description = "Image Generated Successfully"
                            image_found = True
                            break
                
                if not image_found:
                    # Fallback Text from Image Model?
                    ai_description = response.text if hasattr(response, 'text') else "No output data."
                    # If failed to generate pixels but returned text, we use that text. 
                    # But we also try to specifically ask for text description now if completely failed.
                    if not response.parts: 
                        raise ValueError("No candidates or safety block")

            else:
                # Text Model Execution (Fallback)
                response = model.generate_content([f"Describe this new look: {full_prompt}", img])
                ai_description = response.text

        except Exception as gen_err:
            current_app.logger.error(f"Generation Error: {gen_err}")
            ai_description = f"Transformation failed: {str(gen_err)}. (Showing original)"
            # Note: result_image_url remains the original uploaded image as fallback

        # 7. Record Stats
        # (Optional: Add MirrorUsage record here if desired, otherwise omitted for cleanliness)

        return jsonify({
            'status': 'success',
            'result_url': result_image_url,
            'ai_description': ai_description,
            'debug_prompt': full_prompt
        })

    except Exception as e:
        current_app.logger.error(f"Global Error: {e}")
        return jsonify({'error': str(e)}), 500
