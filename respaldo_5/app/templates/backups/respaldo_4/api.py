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
        
        # 1. Construct Dynamic Prompt
        # We build a prompt that references each item
        items_description = ""
        for i, item in enumerate(items):
            items_description += f"- Item {i+1} ({item.get('category', 'Garment')}): Apply this item to the person. ID: {item.get('id')}.\n"
            
        dynamic_prompt = f"""
        ACT AS A VIRTUAL TAILOR AND PROFESSIONAL PHOTO EDITOR.
        
        TASK:
        Generate a realistic "Virtual Try-On" simulation.
        Base Person Image: Provided.
        Clothing Items to Apply:
        {items_description}
        
        CRITICAL RULES FOR "SASTRE VIRTUAL":
        1. BODY FIT (Dynamic Adjustment):
           - Detect shoulder width, waist, and arm length of the person in the Base Image.
           - Adapt the new garments to these EXACT measurements.
           - Preserve natural folds and fabric drape based on the person's pose.
           
        2. TEXTURE CONSISTENCY:
           - If an item is Cotton -> Render with soft, matte texture.
           - If an item is Denim -> Render with heavy, twill texture.
           - If an item is Leather/Silk -> Render with appropriate sheen/reflections.
           
        3. POSITIONING & ANCHORING:
           - Accessories (bags, belts) must be anchored to body points (hips, shoulders).
           - Maintain correct perspective and depth (items must not "float").
           - Glasses must sit correctly on the nose bridge.
           
        4. OUTPUT STYLE:
           - Photorealistic, high resolution.
           - Lighting must match the Base Person Image exactly.
           
        OUTPUT FORMAT:
        Return a text description of the transformation (for now, as image generation is simulated).
        """
        
        print(f"DEBUG: Virtual Tailor Prompt:\n{dynamic_prompt}")
        
        # 2. Call Gemini Service
        # We pass the person image and ALL item images
        item_images = [item['image'] for item in items]
        
        # Use a fixed seed for reproducibility if needed, or random
        # For "Sastre", maybe we want variety? Let's use a fixed seed for stability for now as requested.
        FIXED_SEED = 123456 
        
        result = gemini_service.generate_try_on(person_image, item_images, dynamic_prompt, seed=FIXED_SEED)
        
        if not result['success']:
             return jsonify({'success': False, 'error': result.get('error')}), 500
             
        # 3. Handle Result & Steganography
        # Since Gemini 2.0 Flash returns text description (not image bytes directly yet for this flow),
        # we will SIMULATE the image return by using the ORIGINAL person image but embedding the metadata.
        # In a real production with Imagen 3, we would use the generated image bytes.
        
        description = result['description']
        
        # Decode person image to embed metadata
        if "base64," in person_image:
            header, encoded = person_image.split("base64,", 1)
        else:
            encoded = person_image
            header = "data:image/png;base64,"
            
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Create Rich Metadata (The "Digital Twin")
        meta = PngImagePlugin.PngInfo()
        
        # Construct the JSON Metadata object
        metadata_obj = {
            "imagen_generada": "simulated_overlay", # Placeholder
            "metadata_esteganografico": {
                "prenda_sustituida": [str(item.get('id')) for item in items],
                "ajuste_corporal": "Auto-detected (Gemini Vision)",
                "iluminacion_aplicada": "Matched to Base Image",
                "timestamp": "2025-12-02T12:00:00Z", # Should be dynamic
                "ai_model": "Gemini 2.0 Flash (Tailor Mode)"
            },
            "transformation_description": description
        }
        
        meta.add_text("VirtualTailorData", json.dumps(metadata_obj))
        meta.add_text("Description", description)
        
        # Save
        output = io.BytesIO()
        img.save(output, format="PNG", pnginfo=meta)
        output.seek(0)
        
        processed_base64 = header + base64.b64encode(output.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': processed_base64,
            'description': description,
            'metadata': metadata_obj
        })

    except Exception as e:
        print(f"Error in virtual try-on: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
