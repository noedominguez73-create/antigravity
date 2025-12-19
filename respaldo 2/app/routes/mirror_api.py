from flask import Blueprint, request, jsonify, current_app, render_template
from app import db
from app.models import MirrorItem, MirrorUsage
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime
import google.generativeai as genai
from PIL import Image, PngImagePlugin
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



def record_token_usage(usage_type, prompt_tokens, completion_tokens, total_tokens, item_id=None):
    """Helper to record token usage to database."""
    try:
        # Always record usage, even if tokens are 0
        usage = MirrorUsage(
            usage_type=usage_type,
            item_id=item_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        db.session.add(usage)
        db.session.commit()
        current_app.logger.info(f"Recorded usage: {usage_type} ({total_tokens} tokens)")
    except Exception as e:
        current_app.logger.error(f"Failed to record usage stats: {e}")

def get_best_model(capability='image'):
    """Dynamically select the best available model."""
    try:
        # standard fallback if listing fails or returns nothing useful
        fallback = 'models/gemini-1.5-flash'
        
        all_models = genai.list_models()
        # Filter for generateContent support
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            current_app.logger.warning("No valid models found from API list.")
            return fallback

        selected_model = None
        
        # Simple selection logic based on capability keywords
        if capability == 'image':
            # Prefer 2.0/EXP image models -> then any 'image' model -> then 1.5 Pro/Flash as fallback (some can do image gen)
            candidates = [m for m in valid_models if 'image' in m.lower() or '2.0' in m]
            if candidates: selected_model = candidates[0]
            
        elif capability == 'vision':
            # Prefer 1.5 Pro -> 1.5 Flash -> 2.0 -> Any Flash -> Any Pro
            # Priority list check
            priority_keywords = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0', 'flash']
            for kw in priority_keywords:
                match = next((m for m in valid_models if kw in m), None)
                if match:
                    selected_model = match
                    break
        
        # Final fallback to first valid model if specific search failed
        if not selected_model:
            selected_model = valid_models[0]

        current_app.logger.info(f"Selected Model for {capability}: {selected_model}")
        return selected_model

    except Exception as e:
        current_app.logger.error(f"Error selecting model: {e}")
        return 'models/gemini-1.5-flash'

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

@mirror_api_bp.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.lower().endswith(('.mp4', '.webm')):
        try:
            # Fixed name for simplicity in frontend
            filename = 'tutorial.mp4'
            upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'mirror')
            os.makedirs(upload_folder, exist_ok=True)
            
            save_path = os.path.join(upload_folder, filename)
            file.save(save_path)
            
            # Return URL with timestamp cache buster
            url = f"/static/uploads/mirror/{filename}?t={int(time.time())}"
            return jsonify({'url': url, 'message': 'Video uploaded successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type (MP4/WebM only)'}), 400

@mirror_api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Returns usage statistics."""
    try:
        count = MirrorUsage.query.filter_by(usage_type='generation').count()
        
        # Calculate Token Sums
        # Calculate Generation Tokens
        gen_stats = db.session.query(
            db.func.sum(MirrorUsage.total_tokens)
        ).filter_by(usage_type='generation').first()
        gen_tokens = gen_stats[0] or 0

        # Calculate Analysis Tokens
        analysis_stats = db.session.query(
            db.func.sum(MirrorUsage.total_tokens)
        ).filter_by(usage_type='analysis').first()
        analysis_tokens = analysis_stats[0] or 0

        total_tokens = gen_tokens + analysis_tokens

        return jsonify({
            'generations': count,
            'generation_tokens': gen_tokens,
            'analysis_tokens': analysis_tokens,
            'total_tokens': total_tokens
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching stats: {e}")
        return jsonify({
            'generations': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        })

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
        
        generated_prompt = "Manual override prompt required"
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

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
                    
                    # Extract Tokens
                    if hasattr(response, 'usage_metadata'):
                        prompt_tokens = response.usage_metadata.prompt_token_count
                        completion_tokens = response.usage_metadata.candidates_token_count
                        total_tokens = response.usage_metadata.total_token_count
                else:
                    current_app.logger.warning("No vision model available for upload analysis.")

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
        
        # Record Usage Stats
        record_token_usage('analysis', prompt_tokens, completion_tokens, total_tokens, item_id=new_item.id)

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
        print(f"DEBUG: Input Prompt: {full_prompt}")

        # 5. Select Model
        # Force model based on user request (confirmed 2.5 exists via list)
        model_name = 'models/gemini-2.5-flash-image'
        print(f"DEBUG: Selected Model: {model_name}")

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

            # Extract Token Usage
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count

        except Exception as gen_err:
            current_app.logger.error(f"Generation Error: {gen_err}")
            ai_description = f"Transformation failed: {str(gen_err)}. (Showing original)"
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            # Note: result_image_url remains the original uploaded image as fallback

        # 7. Record Stats
        record_token_usage('generation', prompt_tokens, completion_tokens, total_tokens)

        return jsonify({
            'status': 'success',
            'result_url': result_image_url,
            'ai_description': ai_description,
            'debug_prompt': full_prompt
        })

    except Exception as e:
        current_app.logger.error(f"Global Error: {e}")
        return jsonify({'error': str(e)}), 500
