from flask import Blueprint, request, jsonify
from app.services.gemini_service import gemini_service
import base64
import io
from PIL import Image, PngImagePlugin
import json

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/process-closet-item', methods=['POST'])
def process_closet_item():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
        image_data = data['image']
        category = data.get('category', 'Clothing')
        
        # CONSTANTS FOR QUALITY CONTROL
        INVENTORY_STYLE_PROMPT_BASE = "Transforma este artículo de ropa en una imagen de 512x512 píxeles. Coloca la prenda perfectamente centrada sobre un FONDO BLANCO PURO (#FFFFFF). Aplica un estilo de FLAT DESIGN, elimina todas las sombras, arrugas y texturas complejas para crear un aspecto plano y uniforme. El estilo debe ser el de una foto profesional de inventario de e-commerce. CRITICAL RULES: ONLY change the background and stylization to inventory format. The original garment details must be preserved."
        FIXED_AI_SEED = 123456

        # Construct Final Prompt
        # We combine the category and the base prompt to ensure consistency
        final_prompt = f"""
        Context: {INVENTORY_STYLE_PROMPT_BASE}
        
        Task: Analyze this {category} image based on the style rules above.
        Provide a high-quality product photography description in Spanish.
        
        Format the output exactly like this example:
        
        Fotografía de producto de alta calidad de [Género/Categoría] [Color] [Nombre del Artículo].
        Tejido: [Detalles del material].
        Detalles de diseño: [Cuello, mangas, ajuste, características únicas].
        Estilo: [Descripción del estilo].
        Iluminación profesional de producto, fondo blanco o neutro, estilo de catálogo de ropa, detalles nítidos.
        """
        
        print(f"DEBUG: Sending to AI with Seed {FIXED_AI_SEED}")
        print(f"DEBUG: Final Prompt: {final_prompt}")

        analysis = gemini_service.analyze_image(image_data, final_prompt, seed=FIXED_AI_SEED)
        
        if not analysis['success']:
            return jsonify({'success': False, 'error': analysis.get('error')}), 500
            
        description = analysis['content']
        
        # 2. Steganography / Metadata Embedding
        # We will embed the description into the PNG metadata (tEXt chunk)
        
        # Decode base64 image
        if "base64," in image_data:
            header, encoded = image_data.split("base64,", 1)
        else:
            encoded = image_data
            
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Create metadata
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Description", description)
        meta.add_text("AI_Generator", "Gemini 2.0 Flash")
        meta.add_text("Category", category)
        
        # Save to bytes with metadata
        output = io.BytesIO()
        img.save(output, format="PNG", pnginfo=meta)
        output.seek(0)
        
        # Re-encode to base64
        processed_base64 = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': processed_base64,
            'description': description
        })
        
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/virtual-try-on', methods=['POST'])
def virtual_try_on():
    try:
        data = request.json
        if not data or 'person_image' not in data or 'items' not in data:
            return jsonify({'success': False, 'error': 'Missing person_image or items'}), 400
            
        person_image = data['person_image']
        items = data['items'] # List of {id, category, image}
        client_id = data.get('client_id', 'unknown_client') # Should come from auth
        
        # Initialize Robust Manager
        from app.services.tryon_service import VirtualTryOnManager, TryOnStatus
        manager = VirtualTryOnManager(client_id)
        
        # Execute with Retries & Validation
        result = manager.virtual_tryon(person_image, items)
        
        # ===== BILLING LOGIC (CRITICAL) =====
        # Only charge 1 credit if SUCCESS
        client_charged = 0
        platform_absorbed = 0.0
        
        if result["status"] == TryOnStatus.SUCCESS.value:
            client_charged = 1
            # Calculate absorbed cost (mock cost per attempt)
            cost_per_attempt = 0.05 
            attempts = result.get("attempts", 1)
            platform_absorbed = (attempts - 1) * cost_per_attempt
            
            # TODO: Call actual DB function to deduct credit
            # db.deduct_credit(client_id, 1)
            # db.log_absorbed_cost(client_id, platform_absorbed)
            
            print(f"BILLING: Charged Client {client_charged} credit. Platform absorbed ${platform_absorbed:.2f} (Attempts: {attempts})")

        return jsonify({
            'success': result["status"] == TryOnStatus.SUCCESS.value,
            'image': result.get("image"),
            'description': result.get("description"),
            'status': result["status"],
            'attempts': result.get("attempts"),
            'billing': {
                'client_charged': client_charged,
                'platform_absorbed': platform_absorbed,
                'message': "Solo se cobra si el resultado es exitoso."
            },
            'logs': result.get("logs")
        })

    except Exception as e:
        print(f"Error in virtual try-on: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
