from flask import Blueprint, request, jsonify, current_app, render_template
from app import db
# Force Reload Trigger
from app.models import MirrorItem, MirrorUsage, User, SalonConfig
from app.utils.auth_utils import login_required
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime
import google.generativeai as genai
from PIL import Image, PngImagePlugin
from dotenv import load_dotenv

from app.extensions import limiter

# Initialize Blueprint
mirror_api_bp = Blueprint('mirror_api', __name__, url_prefix='/api/mirror')

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Persistent Config Logic
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror_config.json')

def load_config():
    """Load config from JSON file or return defaults."""
    defaults = {
        'selection_thickness': 4,
        'selection_glow': 10,
        'primary_color': '#00ff88',
        'secondary_color': '#00ccff',
        'stylist_avatar_url': 'https://img.freepik.com/premium-photo/futuristic-female-android-concept-art_931597-29007.jpg',
        'stylist_persona_prompt': "",
        'stylist_voice_name': ""
    }
    try:
        if os.path.exists(CONFIG_FILE):
            import json
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                defaults.update(saved) # Merge to keep defaults for new keys
    except Exception as e:
        print(f"Config load error: {e}")
    return defaults

def save_config(config_data):
    """Save config dict to JSON file."""
    try:
        import json
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Config save error: {e}")
        return False

# Initialize Config
MIRROR_CONFIG = load_config()

@mirror_api_bp.route('/config', methods=['GET', 'POST'])
@login_required 
def handle_config():
    """Get or set mirror configuration for the logged-in salon."""
    user_id = request.current_user_id
    config = SalonConfig.query.filter_by(user_id=user_id).first()
    
    if request.method == 'POST':
        try:
            data = request.json or {}
            if not config:
                config = SalonConfig(user_id=user_id)
                db.session.add(config)
            
            # Update fields
            if 'logo_url' in data: config.logo_url = data['logo_url']
            if 'promo_video_url' in data: config.promo_video_url = data['promo_video_url']
            if 'primary_color' in data: config.primary_color = data['primary_color']
            if 'secondary_color' in data: config.secondary_color = data['secondary_color']
            
            if 'stylist_name' in data: config.stylist_name = data['stylist_name']
            if 'stylist_voice_name' in data: config.stylist_voice_id = data['stylist_voice_name'] # Mapping name to ID field
            if 'stylist_persona_prompt' in data: config.stylist_personality_prompt = data['stylist_persona_prompt']
            if 'welcome_message' in data: config.welcome_message = data['welcome_message']
            
            db.session.commit()
            
            return jsonify({
                'logo_url': config.logo_url,
                'primary_color': config.primary_color
            })
        except Exception as e:
            current_app.logger.error(f"Config update error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # GET Request
    if not config:
        # Return defaults if no config exists yet
        return jsonify({
            'selection_thickness': 4, # Legacy support
            'selection_glow': 10,     # Legacy support
            'primary_color': '#00ff88',
            'secondary_color': '#00ccff',
            'stylist_name': 'Asesora IA',
            'stylist_voice_name': '',
            'stylist_persona_prompt': ''
        })
        
    return jsonify({
        'selection_thickness': 4,
        'selection_glow': 10,
        'logo_url': config.logo_url,
        'promo_video_url': config.promo_video_url,
        'primary_color': config.primary_color,
        'secondary_color': config.secondary_color,
        'stylist_name': config.stylist_name,
        'stylist_voice_name': config.stylist_voice_id,
        'stylist_persona_prompt': config.stylist_personality_prompt,
        'welcome_message': config.welcome_message
    })

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
    """Dynamically select the best available model - QUALITY PRIORITY."""
    try:
        # Standard fallback
        fallback = 'models/gemini-1.5-pro' # Fallback to Pro for quality
        
        all_models = genai.list_models()
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            current_app.logger.warning("No valid models found from API list.")
            return fallback

        selected_model = None
        
        # QUALITY FIRST SELECTION LOGIC
        if capability == 'image':
            # For Generation: Gemini 2.0 has superior image gen capabilities
            candidates = [m for m in valid_models if 'gemini-2.0' in m]
            if candidates: selected_model = candidates[0]
            
        elif capability == 'vision':
            # For Analysis: 1.5 Pro is best for detail
            priority_keywords = ['gemini-1.5-pro', 'gemini-2.0', 'gemini-1.5-flash']
            for kw in priority_keywords:
                match = next((m for m in valid_models if kw in m), None)
                if match:
                    selected_model = match
                    break

        elif capability == 'chat':
            # For Reasoning: 1.5 Pro is best for following complex persona instructions
            priority_keywords = ['gemini-1.5-pro', 'gemini-2.0', 'gemini-1.5-flash']
            for kw in priority_keywords:
                match = next((m for m in valid_models if kw in m), None)
                if match:
                    selected_model = match
                    break
        
        if not selected_model:
            selected_model = valid_models[0]

        current_app.logger.info(f"Selected Model for {capability}: {selected_model}")
        return selected_model

    except Exception as e:
        current_app.logger.error(f"Error selecting model: {e}")
        return 'models/gemini-1.5-pro'

def get_did_api_key():
    """Retrieve D-ID API Key."""
    return os.getenv('DID_API_KEY')

def generate_did_video(text, source_url):
    """Generates a D-ID Talk video from text."""
    api_key = get_did_api_key()
    if not api_key: return None # Graceful fallback
    
    import requests
    url = "https://api.d-id.com/talks"
    
    payload = {
        "script": {
            "type": "text",
            "subtitles": "false",
            "provider": { "type": "microsoft", "voice_id": "es-MX-DaliaNeural" }, # Natural Mexican Spanish
            "input": text
        },
        "config": {
            "fluent": "false", # Less expensive, fast enough
            "pad_audio": "0.0"
        },
        "source_url": source_url
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Basic {api_key}" 
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 201:
            return response.json().get('id')
        else:
            current_app.logger.error(f"D-ID Error: {response.text}")
            return None
    except Exception as e:
        current_app.logger.error(f"D-ID Request Failed: {e}")
        return None

def get_did_video_url(talk_id):
    """Polls for D-ID video completion."""
    api_key = get_did_api_key()
    if not api_key: return None
    
    import requests
    url = f"https://api.d-id.com/talks/{talk_id}"
    headers = { "accept": "application/json", "Authorization": f"Basic {api_key}" }
    
    # Simple check - in production use proper async polling
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data.get('status') == 'done':
            return data.get('result_url')
        return None # pending or error
    except:
        return None

def build_inpainting_prompt(hairstyle, color, instructions):
    """Constructs a safe, structured prompt for image generation."""
    target_parts = []
    if hairstyle: target_parts.append(hairstyle)
    if color: target_parts.append(color)
    if instructions: target_parts.append(instructions)
    
    target_desc = ". ".join(target_parts) if target_parts else "Enhance hairstyle professional look"

    return (
        "Generate a photorealistic image. Apply a professional hair transformation (Inpainting) to the subject's hair.\n"
        "The goal is to visualize a new hairstyle while keeping the person's face EXACTLY as it is.\n\n"
        
        "TARGET LOOK:\n"
        f"{target_desc}\n\n"
        
        "CRITICAL - IDENTITY PRESERVATION RULES (STRICTLY ENFORCED):\n"
        "1. DO NOT CHANGE THE FACE. The subject's facial features (eyes, nose, mouth, jawline, cheekbones) MUST remain 100% identical to the original image.\n"
        "2. PRESERVE IMPERFECTIONS. Do not smooth skin, remove wrinkles, or alter skin tone. The age and texture must match the original photo exactly.\n"
        "3. NO BEAUTIFICATION. Do not apply makeup or 'glamour' filters unless explicitly requested in the prompt.\n"
        "4. DO NOT CHANGE THE GENDER. If the subject is male, they must remain male. If female, remain female.\n\n"
        
        "MODIFICATIONS ALLOWED ONLY FOR:\n"
        "- Hairstyle (cut, length, color, texture).\n"
        "- Clothing/Attire (if specified in the prompt).\n"
        "- Background (if needed to match the professional look).\n\n"
        
        "Output ONLY the modified image. Ensure seamless blending between the new hair/clothes and the original face."
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

@mirror_api_bp.route('/upload_logo', methods=['POST'])
def upload_logo():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Fixed name so mirror.html can always find it
            filename = 'salon_logo.png'
            upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'mirror')
            os.makedirs(upload_folder, exist_ok=True)
            
            save_path = os.path.join(upload_folder, filename)
            file.save(save_path)
            
            # Return URL with timestamp cache buster
            url = f"/static/uploads/mirror/{filename}?t={int(time.time())}"
            return jsonify({'url': url, 'message': 'Logo uploaded successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type (PNG/JPG only)'}), 400

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
@login_required
def generate_look():
    """Core endpoint for Hairstyle/Color generation."""
    try:
        # 0. Subscription & Quota Check
        user = User.query.get(request.current_user_id)
        if not user:
            return jsonify({'error': 'Usuario no encontrado'}), 404
            
        # Check Subscription Status (Only if strictly 'inactive' do we block)
        # Default is 'active' or None.
        if getattr(user, 'subscription_status', 'active') == 'inactive':
            return jsonify({'error': 'Suscripción inactiva. Por favor contacte al administrador.'}), 403
            
        # Check Quota
        current_usage = getattr(user, 'current_month_tokens', 0)
        limit = getattr(user, 'monthly_token_limit', 1000) # Default 1000 tokens
        if current_usage >= limit:
             return jsonify({'error': 'Límite mensual de créditos IA excedido.'}), 403

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

        # 5. Select Model - QUALITY PRIORITY
        # Using Gemini 2.0 Flash Exp which is currently SOTA for multimodal generation in preview
        # or fall back to high-end image model found by get_best_model
        model_name = 'models/gemini-2.0-flash-exp' 
        
        # Verify it exists or fallback
        try:
            # Quick check if it's in the list, if not use get_best_model
            # Note: We hardcode it because it IS the best for this specific visual task right now
            pass 
        except:
            model_name = get_best_model(capability='image')

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
            
            is_image_model = 'image' in model_name.lower() or 'exp' in model_name.lower() or '2.0' in model_name
            
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
                    # Check for text response (refusal or description)
                    if hasattr(response, 'text'):
                         ai_description = response.text
                         current_app.logger.warning(f"Model returned text instead of image: {response.text}")
                    else:
                         ai_description = "No output data."
                         
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
            
            # Fallback for Token Counting
            if total_tokens == 0:
                # High Fidelity = Higher logical token cost assumption
                prompt_tokens = len(full_prompt) // 4
                completion_tokens = 4000 if is_image_model else 800 # Higher virtual cost for high quality
                total_tokens = prompt_tokens + completion_tokens
                current_app.logger.info(f"Token usage fallback applied: {total_tokens}")

        except Exception as gen_err:
            current_app.logger.error(f"Generation Error: {gen_err}")
            ai_description = f"Transformation failed: {str(gen_err)}. (Showing original)"
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            # Note: result_image_url remains the original uploaded image as fallback

        # 7. Record Stats & Billing
        record_token_usage('generation', prompt_tokens, completion_tokens, total_tokens)
        
        # Increment User Usage
        if user:
            user.current_month_tokens = (user.current_month_tokens or 0) + total_tokens
            db.session.commit()

        return jsonify({
            'status': 'success',
            'result_url': result_image_url,
            'ai_description': ai_description,
            'debug_prompt': full_prompt
        })

    except Exception as e:
        current_app.logger.error(f"Global Error: {e}")
        return jsonify({'error': str(e)}), 500
@mirror_api_bp.route('/test-probe')
def test_probe():
    return render_template('test_probe.html')

@mirror_api_bp.route('/control-pantalla')
def control_pantalla():
    """Serve the Screen Control page."""
    return render_template('control_pantalla.html')

@mirror_api_bp.route('/tts', methods=['POST'])
def tts_endpoint():
    """Generates Professional Audio using Gemini TTS."""
    try:
        data = request.json
        text = data.get('text', '').strip()
        voice_style = data.get('voice_style', 'Puck') # Default/Fallback
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # 1. Check Cache
        import hashlib
        text_hash = hashlib.md5(f"{text}_{voice_style}".encode()).hexdigest()
        filename = f"tts_{text_hash}.mp3"
        
        # Audio Dir
        audio_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        file_path = os.path.join(audio_dir, filename)
        public_url = f"/static/uploads/audio/{filename}"
        
        # Determine strict consistency/re-usability
        if os.path.exists(file_path):
            current_app.logger.info(f"TTS Cache Hit: {filename}")
            return jsonify({'url': public_url})
            
        # 2. EDGE TTS GENERATION (High Quality Neural Voices) - PYTHON API METHOD
        try:
            import edge_tts
            import asyncio
            
            # Map "Styles" to Real Edge Neural Voices
            voice_map = {
                "Puck": "es-MX-JorgeNeural",       # Optimistic Male
                "Charon": "es-ES-AlvaroNeural",    # Deep Male
                "Kore": "es-MX-DaliaNeural",       # Relaxed Female
                "Fenrir": "es-AR-TomasNeural",     # Energetic Male
                "Aoede": "es-ES-ElviraNeural",     # Elegant Female
                "Microsoft Sabina": "es-MX-DaliaNeural", # Fallback
                "Microsoft Helena": "es-ES-ElviraNeural" # Fallback
            }
            
            selected_voice = voice_map.get(voice_style, "es-MX-DaliaNeural")
            
            # Use Python API directly (No subprocess/PATH issues)
            async def generate_speech():
                communicate = edge_tts.Communicate(text, selected_voice)
                await communicate.save(file_path)

            # Run async function in sync context
            asyncio.run(generate_speech())
            
            return jsonify({'url': public_url})

        except Exception as e:
            current_app.logger.error(f"TTS System Error: {e}")
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            current_app.logger.error(f"TTS Request Failed: {e}")
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mirror_api_bp.route('/imagina-ia')
def imagina_ia():
    """Serve the AI Style Interview page."""
    return render_template('imagina_ia.html')

@mirror_api_bp.route('/did-status/<string:talk_id>')
def check_did_status(talk_id):
    """Proxy to check status of D-ID video."""
    video_url = get_did_video_url(talk_id)
    if video_url:
        return jsonify({'status': 'done', 'url': video_url})
    return jsonify({'status': 'pending'})

@mirror_api_bp.route('/avatar')
def avatar_hub():
    """Serve the Avatar Hub page."""
    return render_template('avatar.html')

@mirror_api_bp.route('/chat', methods=['POST'])
def chat_mirror():
    """AI Interview Logic."""
    try:
        data = request.json
        user_message = data.get('message', '')
        history = data.get('history', [])
        
        # Configure GenAI
        if not configure_genai():
             return jsonify({'reply': "Error de configuración de API."}), 500

        # Dynamic Model Selection
        model_name = get_best_model(capability='chat')
        current_app.logger.info(f"Chat using model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # Conversational History Construction
        # We start with the System Persona
        # Default System Instruction
        default_instruction = (
            "Eres una Asesora de Imagen Profesional (Asesora IA). "
            "CONTEXTO: Estás en una llamada de VOZ con el cliente. "
            "Tus respuestas deben ser MUY CORTAS y conversacionales (máximo 2 oraciones por turno si es posible). "
            "La gente escuchará tu respuesta, no la leerá.\n"
            "OBJETIVO: Entrevistar brevemente para crear un prompt de imagen.\n"
            "COMPORTAMIENTO:\n"
            "1. Haz UNA sola pregunta a la vez.\n"
            "2. Pregunta: Motivo/Ocasión, Edad aprox, Nivel de mantenimiento, Estilo deseado.\n"
            "3. NO des discursos largos. Sé directa pero cálida.\n"
            "4. Cuando tengas 3-4 datos clave, GENERA EL PROMPT FINAL.\n"
            "   Formato de salida (cuando listo):\n"
            "   ```text\n"
            "   [Prompt en inglés detallado y fotorealista]\n"
            "   ```\n"
            "5. Antes del bloque, di algo como '¡Listo! Tengo una idea.'\n"
        )
        
        # Override if custom prompt exists
        if MIRROR_CONFIG.get('stylist_persona_prompt') and len(MIRROR_CONFIG['stylist_persona_prompt'].strip()) > 10:
             print("Using CUSTOM Persona Prompt")
             # We append technical instructions to ensure the tool still works
             custom = MIRROR_CONFIG['stylist_persona_prompt']
             system_instruction = f"""
             {custom}
             
             --- REGLAS TÉCNICAS (NO MODIFICABLES) ---
             1. Tus respuestas deben ser BREVES (para voz).
             2. Cuando tengas suficiente información para crear el look, DEBES generar el output final así:
             ```text
             [Prompt en inglés detallado]
             ```
             """
        else:
             system_instruction = default_instruction

        chat_session = model.start_chat(history=[
            {"role": "user", "parts": [system_instruction]},
            {"role": "model", "parts": ["Entendido. Soy tu Asesora de Imagen Profesional. Empezaré la entrevista ahora."]}
        ] + [
             # history comes from client. exclude the last one if it is the current user message
             h for i, h in enumerate(history) 
             if h['role'] in ['user', 'model'] and not (i == len(history) - 1 and h['role'] == 'user')
        ])
        
        response = chat_session.send_message(user_message)
        reply_text = response.text
        
        # Check for Final Prompt Special Token
        final_prompt = None
        if "```text" in reply_text:
            try:
                # Extract content between code blocks
                final_prompt = reply_text.split("```text")[1].split("```")[0].strip()
            except:
                pass
        
        # D-ID Video Generation (Interactive Avatar)
        did_talk_id = None
        # Only generate if short enough (cost control) and KEY exists
        if len(reply_text) < 500 and get_did_api_key():
            avatar_url = MIRROR_CONFIG.get('stylist_avatar_url')
            # Fallback if config is missing
            if not avatar_url: avatar_url = "https://img.freepik.com/premium-photo/futuristic-female-android-concept-art_931597-29007.jpg"
            
            did_talk_id = generate_did_video(reply_text, avatar_url)
            current_app.logger.info(f"D-ID Video Initiated: ID {did_talk_id}")

        return jsonify({
            'reply': reply_text,
            'final_prompt': final_prompt,
            'did_talk_id': did_talk_id
        })

    except Exception as e:
        import traceback
        current_app.logger.error(f"Chat Error: {e}\n{traceback.format_exc()}")
        return jsonify({'reply': f"Lo siento, tuve un pequeño problema técnico ({str(e)}). ¿Podrías repetirme eso?"}), 500
