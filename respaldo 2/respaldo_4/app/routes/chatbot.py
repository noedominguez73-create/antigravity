# Chatbot Routes - Gemini Integration

from flask import Blueprint, request, jsonify, current_app
from app import db
from app.models import Professional, ChatbotConfig, ChatMessage, Credit, Service
from app.utils.auth_utils import login_required, professional_required
from app.utils.validators import success_response
from app.utils.error_handler import error_response
from app.extensions import limiter
import uuid
import os

import traceback
from app.services.gemini_service import gemini_service
from app.services.rag_service import rag_service
from app.utils.file_extractor import FileExtractor
from app.utils.web_scraper import WebScraper
from werkzeug.utils import secure_filename
# Crear carpeta de uploads si no existe
UPLOAD_FOLDER = 'uploads/knowledge_base'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

chatbot_bp = Blueprint('chatbot', __name__, url_prefix='/api/chatbot')

@chatbot_bp.route('/<int:profesional_id>/config', methods=['GET'])
def get_config(profesional_id):
    """Get chatbot configuration (public endpoint for widget)"""
    
    # Get professional
    prof = Professional.query.get(profesional_id)
    
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Get chatbot config
    config = ChatbotConfig.query.filter_by(professional_id=profesional_id).first()
    
    if not config:
        return success_response({
            'is_active': False,
            'welcome_message': '',
            'system_prompt': '',
            'knowledge_base': '',
            'temperature': 0.7,
            'max_tokens': 1000,
            'professional_name': prof.user.full_name,
            'specialty': prof.specialty,
            'message': 'Chatbot no configurado para este profesional'
        })
    
    return success_response({
        'is_active': config.is_active,
        'welcome_message': config.welcome_message or '',
        # 'system_prompt': config.system_prompt,  <-- REMOVED FOR SECURITY (VULN-002)
        # 'knowledge_base': config.knowledge_base, <-- REMOVED FOR SECURITY (VULN-002)
        'temperature': config.temperature or 0.7,
        'max_tokens': config.max_tokens or 1000,
        'professional_name': prof.user.full_name,
        'specialty': prof.specialty
    })

@chatbot_bp.route('/<int:profesional_id>/chat', methods=['POST'])
@chatbot_bp.route('/<int:profesional_id>/mensaje', methods=['POST'])
@limiter.limit("10 per minute") # VULN-004 Fix
def send_message(profesional_id):
    """Send message to professional's chatbot"""
    data = request.get_json()
    
    message = data.get('message') or data.get('mensaje')
    session_id = data.get('session_id') or str(uuid.uuid4())
    
    if not message:
        return error_response('Mensaje requerido')
    
    # Get professional
    prof = Professional.query.get(profesional_id)
    if not prof or not prof.is_active:
        return error_response('Profesional no encontrado', 404)
    
    # Check if requester is owner/admin
    is_owner = False
    try:
        from app.utils.auth_utils import get_token_from_header, decode_token
        token = get_token_from_header()
        if token:
            payload = decode_token(token)
            if payload:
                user_id = payload['user_id']
                role = payload['role']
                
                if prof.user_id == user_id:
                    is_owner = True
                if role == 'admin':
                    is_owner = True
    except Exception:
        pass

    # Get chatbot config
    config = ChatbotConfig.query.filter_by(professional_id=profesional_id).first()
    
    if (not config or not config.is_active) and not is_owner:
        return error_response('Chatbot no disponible para este profesional', 404)
    
    # Check credits (bypass for owner/admin)
    available_credits = 9999 # Default for owner/admin
    if not is_owner:
        try:
            # ---------------------------------------------------------
            # ATOMIC CREDIT CHECK (VULN-005 FIX)
            # ---------------------------------------------------------
            # Lock the professional row for update to prevent race conditions
            prof_locked = db.session.query(Professional).with_for_update().get(profesional_id)
            
            if not prof_locked:
                return error_response('Profesional no encontrado', 404)

            # Check balance directly from the ledger column
            available_credits = prof_locked.balance
            
            if available_credits < 1:
                return error_response('Créditos insuficientes. Por favor recarga tu cuenta.', 402)
                
            # Deduct credit ATOMICALLY (Pending commit at end of request)
            # We deduct 1 credit tentatively. If cache hit, we might refund or not deduct.
            # Strategy: Deduct now, commit later.
            prof_locked.balance -= 1
            db.session.add(prof_locked)
            # Do NOT commit yet. We commit after successful response generation.
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Database Error during credit lock: {e}")
            return error_response('Error procesando transacción de créditos', 500)
    
    # Save user message
    user_message = ChatMessage(
        professional_id=profesional_id,
        session_id=session_id,
        role='user',
        content=message,
        credits_used=0
    )
    db.session.add(user_message)
    
    # Get services
    services = Service.query.filter_by(professional_id=profesional_id).all()
    services_list = [f"{s.name}: {s.description}" if s.description else s.name for s in services]
    services_text = "\n- ".join(services_list) if services_list else "servicios profesionales de calidad"
    
    # Use configured welcome message
    welcome_msg = config.welcome_message if config and config.welcome_message else f"Hola! Soy {prof.user.full_name}"
    
    # Build context for Gemini
    system_prompt = config.system_prompt if config and config.system_prompt else f"Eres un asistente útil para el profesional {prof.user.full_name}."
    knowledge_base = config.knowledge_base if config and config.knowledge_base else ""
    
    # NUEVO: RAG Context Retrieval (Optimized)
    # Instead of loading all docs, we fetch relevant chunks
    rag_context = rag_service.retrieve_context(message, profesional_id)
    
    documents_context = ""
    if rag_context:
        documents_context = f"\n\n=== INFORMACIÓN RELEVANTE (RAG) ===\n{rag_context}\n"
    else:
        # Fallback: If no RAG context, try legacy full load (optional, maybe skip to save tokens)
        # For now, we keep it empty to enforce RAG usage and save costs.
        pass

    # NUEVO: Obtener URLs del profesional (Keep this as is for now, or RAG it too later)
    # Legacy full-load removed in favor of RAG
    # kb_documents query removed

    # NUEVO: Obtener URLs del profesional
    from app.models import ProfessionalURL
    from datetime import datetime, timedelta
    
    prof_urls = ProfessionalURL.query.filter_by(
        professional_id=profesional_id,
        is_active=True
    ).all()
    
    with open('debug_chatbot.txt', 'a') as f:
        f.write(f"DEBUG: Found {len(prof_urls)} URLs for prof {profesional_id}\n")
    
    urls_context = ""
    if prof_urls:
        urls_context = "\n\n=== CONTENIDO EXTERNO (URLS) ===\n"
        for p_url in prof_urls:
            # Check cache (1 hour)
            should_scrape = False
            if not p_url.cached_content or not p_url.last_fetched:
                should_scrape = True
            else:
                # If older than 1 hour
                if datetime.utcnow() - p_url.last_fetched > timedelta(hours=1):
                    should_scrape = True
            
            with open('debug_chatbot.txt', 'a') as f:
                f.write(f"DEBUG: URL {p_url.url}, should_scrape={should_scrape}\n")
            
            content = p_url.cached_content
            
            if should_scrape:
                with open('debug_chatbot.txt', 'a') as f:
                    f.write(f"Scraping URL: {p_url.url}\n")
                
                scrape_result = WebScraper.fetch_and_clean(p_url.url)
                
                if scrape_result['success']:
                    content = scrape_result['content']
                    # Update DB
                    p_url.cached_content = content
                    p_url.last_fetched = datetime.utcnow()
                    db.session.commit()
                else:
                    with open('debug_chatbot.txt', 'a') as f:
                        f.write(f"Failed to scrape {p_url.url}: {scrape_result.get('error')}\n")
            
            if content:
                urls_context += f"\nFuente: {p_url.url}\n{content[:3000]}\n"

    # Construir contexto de documentos
    # Append URL context to RAG context
    documents_context += urls_context

    # Construct full prompt
    # Construct optimized prompt
    system_instruction = f"""
    Eres el asistente virtual de {prof.user.full_name}. Tu objetivo es responder dudas sobre trámites IMSS basándote EXCLUSIVAMENTE en el contexto proporcionado.
    
    Reglas:
    1. TU ÚNICA FUENTE DE VERDAD es el contexto proporcionado (Documentos y URLs).
    2. NO INVENTES información. Si la respuesta no está explícitamente en el contexto, responde: "Lamento no tener esa información específica. Por favor, contacte directamente a {prof.user.full_name}."
    3. TU ROL es ser el asistente virtual de {prof.user.full_name}. No respondas sobre temas ajenos a sus servicios profesionales.
    4. Sé conciso y profesional.
    5. Formato de respuesta: Markdown.
    
    Instrucciones Personalizadas del Profesional:
    {system_prompt}
    """

    user_prompt = f"""
    === CONTEXTO Y BASE DE CONOCIMIENTO ===
    {knowledge_base}
    
    {documents_context}

    === SERVICIOS OFRECIDOS ===
    {services_text}

    === MENSAJE DEL USUARIO ===
    {message}
    """
    
    # Combine for current API implementation (until system_instruction param is supported in gemini_service)
    full_prompt = f"{system_instruction}\n\n{user_prompt}"

    # ---------------------------------------------------------
    # SEMANTIC CACHE CHECK
    # ---------------------------------------------------------
    from app.services.cache_service import cache_service
    
    # Try to get from cache first
    cached_response = cache_service.get_cached_response(message, profesional_id)
    
    if cached_response:
        # CACHE HIT
        assistant_content = cached_response
        credits_to_deduct = 0 # Optional: Charge 0 or reduced credits for cached hits?
        # Let's charge 0 for now to incentivize efficiency, or 1 if business model requires.
        # Decision: Charge 0 to demonstrate value of AI optimization to user.
        
        # Save assistant message (marked as cached)
        assistant_message = ChatMessage(
            professional_id=profesional_id,
            session_id=session_id,
            role='assistant',
            content=assistant_content,
            credits_used=credits_to_deduct 
        )
        db.session.add(assistant_message)
        db.session.commit()
        
        return success_response({
            'session_id': session_id,
            'respuesta': assistant_content,
            'message': assistant_content,
            'credits_remaining': available_credits - credits_to_deduct,
            'cached': True # Frontend can show "⚡" icon
        })

    # ---------------------------------------------------------
    # GEMINI API CALL (CACHE MISS)
    # ---------------------------------------------------------

    try:
        # Call Gemini API
        analysis = gemini_service.analyze_inquiry(full_prompt)
        
        if analysis['success']:
            assistant_content = analysis['content']
            
            # SAVE TO CACHE
            cache_service.add_to_cache(message, assistant_content, profesional_id)
            
        else:
            # Fallback in case of API error
            current_app.logger.error(f"Gemini API Error: {analysis.get('error')}")
            assistant_content = f"{welcome_msg}. Lo siento, en este momento no puedo procesar tu solicitud inteligente. Mis servicios incluyen:\n- {services_text}"
            
    except Exception as e:
        current_app.logger.error(f"Error calling Gemini service: {str(e)}")
        assistant_content = f"{welcome_msg}. Disculpa, tuve un problema técnico. ¿Podrías intentar de nuevo?"
    
    # Save assistant message
    assistant_message = ChatMessage(
        professional_id=profesional_id,
        session_id=session_id,
        role='assistant',
        content=assistant_content,
        credits_used=1
    )
    db.session.add(assistant_message)
    
    try:
        db.session.commit()
        return success_response({
            'session_id': session_id,
            'respuesta': assistant_content,
            'message': assistant_content,
            'credits_remaining': available_credits - 1,
            'cached': False
        })
    except Exception as e:
        db.session.rollback()
        try:
            with open('debug_chatbot.txt', 'a') as f:
                f.write(f"CRITICAL ERROR in send_message: {str(e)}\n")
        except:
            pass
        return error_response(f'Error: {str(e)}', 500)

@chatbot_bp.route('/<int:profesional_id>/config', methods=['PUT'])
@professional_required
def update_config(profesional_id):
    """Update chatbot configuration"""
    prof = Professional.query.get(profesional_id)
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    if prof.user_id != request.current_user_id and request.current_user_role != 'admin':
        return error_response('No tienes permiso para editar esta configuración', 403)
    
    data = request.get_json()
    config = ChatbotConfig.query.filter_by(professional_id=profesional_id).first()
    
    if not config:
        config = ChatbotConfig(professional_id=profesional_id)
        db.session.add(config)
    
    # Update fields
    if 'is_active' in data:
        config.is_active = data['is_active']
    if 'welcome_message' in data:
        config.welcome_message = data['welcome_message']
    if 'system_prompt' in data:
        config.system_prompt = data['system_prompt']
    if 'knowledge_base' in data:
        config.knowledge_base = data['knowledge_base']
    if 'max_tokens' in data:
        config.max_tokens = data['max_tokens']
    if 'temperature' in data:
        config.temperature = data['temperature']
    
    try:
        db.session.commit()
        return success_response({
            'config': {
                'is_active': config.is_active,
                'welcome_message': config.welcome_message,
                'system_prompt': config.system_prompt,
                'knowledge_base': config.knowledge_base,
                'max_tokens': config.max_tokens,
                'temperature': config.temperature
            }
        }, 'Configuración guardada exitosamente')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al guardar configuración: {str(e)}', 500)

@chatbot_bp.route('/<int:profesional_id>/subir-documento', methods=['POST'])
@professional_required
def upload_document(profesional_id):
    """Upload knowledge base document"""
    prof = Professional.query.get(profesional_id)
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    if prof.user_id != request.current_user_id and request.current_user_role != 'admin':
        return error_response('No tienes permiso para subir documentos', 403)
    
    data = request.get_json()
    document_text = data.get('document_text')
    
    if not document_text:
        return error_response('Texto del documento requerido')
    
    config = ChatbotConfig.query.filter_by(professional_id=profesional_id).first()
    
    if not config:
        config = ChatbotConfig(professional_id=profesional_id)
        db.session.add(config)
    
    if config.knowledge_base:
        config.knowledge_base += f"\n\n{document_text}"
    else:
        config.knowledge_base = document_text
    
    try:
        db.session.commit()
        return success_response(None, 'Documento agregado a la base de conocimiento')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al subir documento: {str(e)}', 500)

# ═════════════════════════════════════════════════════════
# ENDPOINT 1: SUBIR ARCHIVO A BASE DE CONOCIMIENTO
# ═════════════════════════════════════════════════════════

@chatbot_bp.route('/<int:professional_id>/knowledge-base/upload', methods=['POST'])
@login_required
def upload_knowledge_document(professional_id):
    print("EXECUTING CHATBOT_V2 UPLOAD")
    """
    POST /api/chatbot/{id}/knowledge-base/upload
    
    Subir archivo (PDF, Excel, etc.) a la Base de Conocimiento
    """
    try:
        from app.models import KnowledgeBaseDocument
        
        # Verificar que profesional existe y pertenece al usuario
        prof = Professional.query.get(professional_id)
        if not prof:
            return error_response("Profesional no encontrado", 404)
            
        if prof.user_id != request.current_user_id:
            return error_response("No tienes permiso", 403)
        
        # Verificar que hay archivo
        if 'file' not in request.files:
            return error_response("No se envió archivo", 400)
        
        file = request.files['file']
        
        if file.filename == '':
            return error_response("Archivo vacío", 400)
        
        # Validar tipo de archivo
        if not FileExtractor.allowed_file(file.filename):
            return error_response(
                "Tipo de archivo no permitido. Soportados: txt, pdf, xlsx, docx",
                400
            )
        
        # Validar tamaño
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > FileExtractor.MAX_FILE_SIZE:
            return error_response("Archivo demasiado grande (máx 5MB)", 400)
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        unique_filename = f"{professional_id}_{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(file_path)
        
        # Extraer texto
        file_type = filename.rsplit('.', 1)[1].lower()
        text_content = FileExtractor.extract_text(file_path, file_type)
        
        # Guardar en BD
        doc = KnowledgeBaseDocument(
            professional_id=professional_id,
            filename=unique_filename,
            original_filename=filename,
            file_type=file_type,
            text_content=text_content,
            file_size=file_size
        )
        db.session.add(doc)
        db.session.commit()
        
        current_app.logger.info(
            f"Document uploaded: {filename} (prof: {professional_id})"
        )
        
        return success_response(
            {'document': doc.to_dict()},
            "Archivo subido correctamente",
            201
        )
    
    except Exception as e:
        return error_response(
            f"Error subiendo archivo: {str(e)}",
            500,
            internal_error=e
        )
# ═════════════════════════════════════════════════════════

@chatbot_bp.route('/<int:professional_id>/knowledge-base/documents', methods=['GET'])
@login_required
def get_knowledge_documents(professional_id):
    """
    GET /api/chatbot/{id}/knowledge-base/documents
    
    Listar todos los documentos subidos a la Base de Conocimiento
    """
    try:
        from app.models import KnowledgeBaseDocument

        prof = Professional.query.get(professional_id)
        if not prof:
            return error_response("Profesional no encontrado", 404)
            
        if prof.user_id != request.current_user_id:
            return error_response("No tienes permiso", 403)
        
        documents = KnowledgeBaseDocument.query.filter_by(
            professional_id=professional_id
        ).order_by(KnowledgeBaseDocument.uploaded_at.desc()).all()
        
        return success_response({
            'documents': [doc.to_dict() for doc in documents],
            'total': len(documents)
        })
    
    except Exception as e:
        return error_response(
            "Error listando documentos",
            500,
            internal_error=e
        )

# ═════════════════════════════════════════════════════════
# ENDPOINT 3: ELIMINAR DOCUMENTO
# ═════════════════════════════════════════════════════════

@chatbot_bp.route(
    '/<int:professional_id>/knowledge-base/documents/<int:doc_id>',
    methods=['DELETE']
)
@login_required
def delete_knowledge_document(professional_id, doc_id):
    """
    DELETE /api/chatbot/{id}/knowledge-base/documents/{doc_id}
    
    Eliminar un documento de la Base de Conocimiento
    """
    try:
        from app.models import KnowledgeBaseDocument

        prof = Professional.query.get(professional_id)
        if not prof:
            return error_response("Profesional no encontrado", 404)
            
        if prof.user_id != request.current_user_id:
            return error_response("No tienes permiso", 403)
        
        doc = KnowledgeBaseDocument.query.filter_by(
            id=doc_id,
            professional_id=professional_id
        ).first()
        
        if not doc:
            return error_response("Documento no encontrado", 404)
        
        # Eliminar archivo del servidor
        file_path = os.path.join(UPLOAD_FOLDER, doc.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Eliminar de BD
        db.session.delete(doc)
        db.session.commit()
        
        current_app.logger.info(f"Document deleted: {doc.filename}")
        
        return success_response(None, "Documento eliminado")
    
    except Exception as e:
        return error_response(
            "Error eliminando documento",
            500,
            internal_error=e
        )

@chatbot_bp.route('/<int:profesional_id>/historial', methods=['GET'])
@professional_required
def get_history(profesional_id):
    """Get chat history"""
    prof = Professional.query.get(profesional_id)
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    if prof.user_id != request.current_user_id and request.current_user_role != 'admin':
        return error_response('No tienes permiso para ver este historial', 403)
    
    session_id = request.args.get('session_id')
    limit = request.args.get('limit', 50, type=int)
    
    query = ChatMessage.query.filter_by(professional_id=profesional_id)
    
    if session_id:
        query = query.filter_by(session_id=session_id)
    
    messages = query.order_by(ChatMessage.created_at.desc()).limit(limit).all()
    
    history = []
    for msg in messages:
        history.append({
            'id': msg.id,
            'session_id': msg.session_id,
            'role': msg.role,
            'content': msg.content,
            'credits_used': msg.credits_used,
            'created_at': msg.created_at.isoformat()
        })
    
    return success_response({'messages': history})

@chatbot_bp.route('/<int:profesional_id>/public-history', methods=['GET'])
def get_public_history(profesional_id):
    """Get chat history for public users (requires session_id)"""
    session_id = request.args.get('session_id')
    if not session_id:
        return error_response('Session ID requerido', 400)
        
    limit = request.args.get('limit', 50, type=int)
    
    # Verify professional exists
    prof = Professional.query.get(profesional_id)
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Fetch messages for this session only
    messages = ChatMessage.query.filter_by(
        professional_id=profesional_id,
        session_id=session_id
    ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
    
    history = []
    for msg in messages:
        history.append({
            'id': msg.id,
            'role': msg.role,
            'content': msg.content,
            'created_at': msg.created_at.isoformat()
        })
    
    return success_response({'messages': history})
