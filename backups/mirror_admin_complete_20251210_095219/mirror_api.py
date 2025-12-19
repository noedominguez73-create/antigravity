from flask import Blueprint, request, jsonify, current_app, render_template
from app import db
from app.models import MirrorItem, MirrorUsage
from werkzeug.utils import secure_filename
import os
from datetime import datetime

mirror_api_bp = Blueprint('mirror_api', __name__, url_prefix='/api/mirror')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@mirror_api_bp.route('/admin')
def admin_page():
    return render_template('admin_mirror.html')

@mirror_api_bp.route('/items', methods=['GET'])
def get_items():
    category = request.args.get('category')
    query = MirrorItem.query.filter_by(is_active=True)
    if category:
        query = query.filter_by(category=category)
    
    # Sort by order_index (custom) then created_at (newest first)
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

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Unique filename to prevent overwrite
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        
        upload_folder = os.path.join(current_app.root_path, 'static/uploads/mirror')
        os.makedirs(upload_folder, exist_ok=True)
        
        file_path = os.path.join(upload_folder, filename)
        
        
        file_path = os.path.join(upload_folder, filename)
        
        # 1. Resize and Save Image standard size
        try:
            from PIL import Image
            import google.generativeai as genai
            from dotenv import load_dotenv
            
            # Explicitly load from project root
            # current_app.root_path points to .../app
            project_root = os.path.dirname(current_app.root_path)
            env_path = os.path.join(project_root, '.env')
            load_dotenv(env_path)
            
            img = Image.open(file)
            img = img.convert('RGB')
            # Resize to max 1024x1024 keeping aspect ratio, or exact 512?
            # User asked for "exactamente siempre del mismo tamaño y calidad"
            # Let's standardize to 512x512 for consistency in UI and AI
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            img.save(file_path, quality=95)
            
            # 2. Generate Prompt with Gemini
            api_key = os.getenv('GEMINI_API_KEY')
            generated_prompt = "No prompt generated (API Key missing)"
            
            if api_key:
                genai.configure(api_key=api_key)

                # Define distinct prompts based on category
                if category == 'color':
                    # User-provided color techniques and simplified list
                    color_knowledge = """
                    TÉCNICAS DE COLORACIÓN QUE DEBES RECONOCER:
                    - Rayitos / Highlights (Clásicos, gruesos, finos, babylights, 3D, puntas, face-framing, peek-a-boo)
                    - Balayage (Clásico, californiano, colorido, castaño, rubio)
                    - Ombre (Tradicional, invertido, colorido, rubio, chocolate)
                    - Degradé / Shadow Root (Clásico, sombra de raíz, suave, ahumado)
                    - Coloración Sólida (Rubios: platinado, hielo, miel, dorado; Castaños: chocolate, dorado; Rojos: borgoña, cobrizo; Negro: azabache)
                    - Paneles/Secciones (Money pieces, peek-a-boo, bicolor, split)
                    - Creativas (Galaxy, mermaid, sunset, tie-dye, chrome)
                    
                    ESTRUCTURA OBLIGATORIA DEL PROMPT:
                    1. TÉCNICA: (ej. balayage, babylights, ombre)
                    2. TONOS: (ej. caramelo, miel, rubio ceniza, rojo rubí)
                    3. COLOCACIÓN: (ej. enmarcando el rostro, solo puntas, capas ocultas, raíz difuminada)
                    4. EFECTO: (ej. natural, vibrante, multidimensional, contraste alto)
                    5. INTENSIDAD: (ej. sutil, moderado, audaz)
                    """
                    
                    context_prompt = (
                        "ACTÚA COMO UN EXPERTO COLORISTA DE CABELLO. "
                        "Tu objetivo es analizar la imagen y extraer ÚNICAMENTE la información del COLOR del cabello. "
                        "IGNORA la forma del corte, el largo o el estilo del peinado. Céntrate SOLO en el color. "
                        f"Usa el siguiente vocabulario técnico como referencia: {color_knowledge} "
                        "Genera una descripción profesional siguiendo la ESTRUCTURA OBLIGATORIA. "
                        "Ejemplo esperado: 'Balayage caramelizado: rayitos babylights dorados y caramelo finamente entretejidos, concentrados alrededor del rostro (face-framing) y en las puntas, sobre base de rubio oscuro. Efecto multidimensional y besado por el sol.' "
                        "NO empieces con 'La imagen muestra...' o 'Un peinado...', ve directo a la descripción del color."
                    )
                else:
                    # Default: Hairstyle (Focus on cut and style)
                    context_prompt = (
                        "Act as an expert hair designer and AI prompt engineer. "
                        "Analyze this image and describe the HAIRSTYLE (cut and shape) in extreme detail for use in an 8K photorealistic image generator. "
                        "Focus on: precision of the cut (e.g., french bob, pixie, layers, shag), texture (silky, voluminous, wavy, curly), length, and silhouette. "
                        "You may mention the color briefly as context, but the PRIMARY FOCUS must be the physical cut and style. "
                        "Be concise but professional. Output ONLY the descriptive prompt text."
                    )
                
                # Dynamic Model Discovery Strategy
                models_to_try = []
                try:
                    all_models = genai.list_models()
                    valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
                    
                    # Prioritize best vision models
                    # 1. Gemini 1.5 Pro (Best quality)
                    pro_models = [m for m in valid_models if 'gemini-1.5-pro' in m]
                    # 2. Gemini 1.5 Flash (Fast)
                    flash_models = [m for m in valid_models if 'gemini-1.5-flash' in m]
                    # 3. Legacy Vision
                    vision_models = [m for m in valid_models if 'gemini-pro-vision' in m]
                    # 4. Any other Gemini
                    other_models = [m for m in valid_models if 'gemini' in m and m not in pro_models + flash_models + vision_models]
                    
                    # Construct priority list (take the newest/first of each category)
                    models_to_try.extend(pro_models[:1])
                    models_to_try.extend(flash_models[:1])
                    models_to_try.extend(vision_models[:1])
                    models_to_try.extend(other_models[:1])
                    
                    if not models_to_try:
                        models_to_try = ['models/gemini-1.5-flash-latest', 'models/gemini-1.5-pro-latest']
                        
                except Exception as list_err:
                    current_app.logger.warning(f"Error listing models: {list_err}")
                    models_to_try = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro']
                
                errors = []
                generated_prompt = ""
                
                for model_name in models_to_try:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content([context_prompt, img])
                        generated_prompt = response.text.strip()
                        break # Stop if success
                    except Exception as loop_error:
                        current_app.logger.warning(f"Model {model_name} failed: {loop_error}")
                        errors.append(f"[{model_name}]: {str(loop_error)}")
                        continue
                
                if not generated_prompt:
                    current_app.logger.error("All Gemini models failed.")
                    # In production we might not want authentication errors exposed to client, 
                    # but for this user it was helpful. I will keep it brief.
                    generated_prompt = f"Failed to generate prompt. (See server logs for details)."
        except Exception as e:
            current_app.logger.error(f"Error processing image/gemini: {e}")
            # Fallback save if PIL fails (unlikely)
            file.seek(0)
            file.save(file_path)
            generated_prompt = f"Error generating prompt: {str(e)}"
        
        image_url = f"/static/uploads/mirror/{filename}"
        
        new_item = MirrorItem(
            name=name, 
            category=category, 
            image_url=image_url,
            prompt=generated_prompt,
            order_index=order
        )
        # If color, maybe extract color? For now just manual
        if category == 'color':
            # logic to get color code if provided
            pass
            
        db.session.add(new_item)
        db.session.commit()
        
        return jsonify(new_item.to_dict()), 201

    return jsonify({'error': 'Invalid file type'}), 400

@mirror_api_bp.route('/items/<int:id>', methods=['PUT'])
def update_item(id):
    item = MirrorItem.query.get_or_404(id)
    data = request.json
    
    if 'order_index' in data:
        item.order_index = int(data['order_index'])
    
    if 'name' in data:
        item.name = data['name']
        
    db.session.commit()
    return jsonify(item.to_dict())

@mirror_api_bp.route('/items/<int:id>', methods=['DELETE'])
def delete_item(id):
    item = MirrorItem.query.get_or_404(id)
    item.is_active = False # Soft delete
    db.session.commit()
    return jsonify({'message': 'Item deleted'})

@mirror_api_bp.route('/stats', methods=['GET'])
def get_stats():
    # Mock stats for now or real
    current_month = datetime.now().strftime('%B')
    year = datetime.now().year
    
    # Simple count if usage not implemented fully
    count = MirrorUsage.query.count()
    
    return jsonify({
        'month': current_month,
        'year': year,
        'generations': count,
        'status': 'active'
    })
