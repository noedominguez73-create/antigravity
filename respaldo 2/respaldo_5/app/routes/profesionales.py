# Professional Routes - Profile Management

from flask import Blueprint, request, jsonify
from app import db
from app.models import Professional, User, Service, Experience, Certification, Comment, ProfessionalURL
from app.utils.auth_utils import login_required, professional_required
from app.utils.validators import success_response
from app.utils.error_handler import error_response
from app.utils.web_scraper import WebScraper
from datetime import datetime
from sqlalchemy import or_

profesionales_bp = Blueprint('profesionales', __name__, url_prefix='/api/profesionales')

@profesionales_bp.route('', methods=['GET'])
def get_profesionales():
    """Get list of professionals with optional filters"""
    # Get query parameters
    specialty = request.args.get('especialidad')
    city = request.args.get('ciudad')
    min_rating = request.args.get('calificacion')
    user_id = request.args.get('user_id', type=int)
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Build query
    query = Professional.query.filter_by(is_active=True)
    
    if specialty:
        query = query.filter(Professional.specialty.ilike(f'%{specialty}%'))
    
    if city:
        query = query.filter(Professional.city.ilike(f'%{city}%'))
        
    if user_id:
        query = query.filter(Professional.user_id == user_id)
    
    if min_rating:
        try:
            min_rating = float(min_rating)
            query = query.filter(Professional.rating >= min_rating)
        except ValueError:
            pass
    
    # Order by rating
    query = query.order_by(Professional.rating.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    professionals = []
    for prof in pagination.items:
        professionals.append({
            'id': prof.id,
            'user_id': prof.user_id,
            'full_name': prof.user.full_name,
            'specialty': prof.specialty,
            'city': prof.city,
            'bio': prof.bio,
            'rating': prof.rating,
            'total_reviews': prof.total_reviews,
            'profile_image': prof.profile_image
        })
    
    return success_response({
        'professionals': professionals,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })

@profesionales_bp.route('/<int:id>', methods=['GET'])
def get_profesional(id):
    """Get full professional profile"""
    prof = Professional.query.get(id)
    
    if not prof or not prof.is_active:
        return error_response('Profesional no encontrado', 404)
    
    return success_response({
        'id': prof.id,
        'user_id': prof.user_id,
        'full_name': prof.user.full_name,
        'email': prof.user.email,
        'specialty': prof.specialty,
        'city': prof.city,
        'bio': prof.bio,
        'rating': prof.rating,
        'total_reviews': prof.total_reviews,
        'profile_image': prof.profile_image,
        'created_at': prof.created_at.isoformat()
    })

@profesionales_bp.route('', methods=['POST'])
@login_required
def create_profesional():
    """Create professional profile (authenticated users only)"""
    data = request.get_json()
    
    # Check if user already has a professional profile
    existing = Professional.query.filter_by(user_id=request.current_user_id).first()
    if existing:
        return error_response('Ya tienes un perfil profesional', 409)
    
    # Update user role to professional
    user = User.query.get(request.current_user_id)
    user.role = 'professional'
    
    # Create professional profile
    try:
        prof = Professional(
            user_id=request.current_user_id,
            specialty=data.get('specialty'),
            city=data.get('city'),
            bio=data.get('bio'),
            profile_image=data.get('profile_image')
        )
        db.session.add(prof)
        db.session.commit()
        
        return success_response({
            'id': prof.id,
            'specialty': prof.specialty,
            'city': prof.city
        }, 'Perfil profesional creado exitosamente', 201)
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al crear perfil: {str(e)}', 500)

@profesionales_bp.route('/<int:id>', methods=['PUT'])
@professional_required
def update_profesional(id):
    """Update professional profile (owner only)"""
    prof = Professional.query.get(id)
    
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Check ownership
    # Ensure comparison is safe (str vs int)
    if str(prof.user_id) != str(request.current_user_id) and request.current_user_role != 'admin':
        return error_response(f'No tienes permiso para editar este perfil. (Tu ID: {request.current_user_id}, Dueño: {prof.user_id})', 403)
    
    data = request.get_json()
    
    # Update fields
    if 'specialty' in data:
        prof.specialty = data['specialty']
    if 'city' in data:
        prof.city = data['city']
    if 'bio' in data:
        prof.bio = data['bio']
    if 'profile_image' in data:
        prof.profile_image = data['profile_image']
    
    try:
        db.session.commit()
        return success_response(None, 'Perfil actualizado exitosamente')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al actualizar perfil: {str(e)}', 500)

@profesionales_bp.route('/<int:id>', methods=['DELETE'])
@professional_required
def delete_profesional(id):
    """Soft delete professional profile"""
    prof = Professional.query.get(id)
    
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Check ownership
    if prof.user_id != request.current_user_id and request.current_user_role != 'admin':
        return error_response('No tienes permiso para eliminar este perfil', 403)
    
    # Soft delete
    prof.is_active = False
    
    try:
        db.session.commit()
        return success_response(None, 'Perfil desactivado exitosamente')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al desactivar perfil: {str(e)}', 500)

@profesionales_bp.route('/<int:id>/servicios', methods=['GET'])
def get_servicios(id):
    """Get services offered by professional"""
    prof = Professional.query.get(id)
    
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    services = []
    for service in prof.services:
        services.append({
            'id': service.id,
            'name': service.name,
            'description': service.description,
            'price': service.price,
            'duration_minutes': service.duration_minutes
        })
    
    return success_response({'services': services})

@profesionales_bp.route('/<int:id>/experiencia', methods=['GET'])
def get_experiencia(id):
    """Get work experience of professional"""
    prof = Professional.query.get(id)
    
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    experiences = []
    for exp in prof.experiences:
        experiences.append({
            'id': exp.id,
            'company': exp.company,
            'position': exp.position,
            'description': exp.description,
            'start_date': exp.start_date.isoformat() if exp.start_date else None,
            'end_date': exp.end_date.isoformat() if exp.end_date else None,
            'is_current': exp.is_current
        })
    
    return success_response({'experiences': experiences})

@profesionales_bp.route('/<int:id>/certificaciones', methods=['GET'])
def get_certificaciones(id):
    """Get certifications of professional"""
    prof = Professional.query.get(id)
    
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    certifications = []
    for cert in prof.certifications:
        certifications.append({
            'id': cert.id,
            'name': cert.name,
            'issuer': cert.issuer,
            'issue_date': cert.issue_date.isoformat() if cert.issue_date else None,
            'expiry_date': cert.expiry_date.isoformat() if cert.expiry_date else None,
            'credential_id': cert.credential_id,
            'credential_url': cert.credential_url
        })
    
    return success_response({'certifications': certifications})

@profesionales_bp.route('/<int:id>/comentarios', methods=['GET'])
def get_comentarios_profesional(id):
    """Get approved comments for professional"""
    prof = Professional.query.get(id)
    
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Only show approved comments
    comments = Comment.query.filter_by(
        professional_id=id,
        status='approved'
    ).order_by(Comment.created_at.desc()).all()
    
    comments_list = []
    for comment in comments:
        comments_list.append({
            'id': comment.id,
            'author': comment.author.full_name,
            'rating': comment.rating,
            'content': comment.content,
            'created_at': comment.created_at.isoformat()
        })
    
    return success_response({'comments': comments_list})

@profesionales_bp.route('/<int:id>/posts', methods=['GET'])
def get_posts(id):
    """Get blog posts/content by professional (PLACEHOLDER)"""
    # TODO: Implement blog/posts functionality
    return success_response({'posts': []}, 'Funcionalidad de posts en desarrollo')

# ═════════════════════════════════════════════════════════
# ENDPOINTS DE GESTIÓN DE URLs (AUDITORÍA FASE 4)
# ═════════════════════════════════════════════════════════

@profesionales_bp.route('/<int:professional_id>/urls', methods=['POST'])
@login_required
def add_url(professional_id):
    """
    POST /api/profesionales/{id}/urls
    Agregar nueva URL de referencia
    """
    try:
        # 1. Verificar permisos
        prof = Professional.query.get(professional_id)
        if not prof:
            return error_response("Profesional no encontrado", 404)
            
        if prof.user_id != request.current_user_id:
            return error_response("No tienes permiso", 403)
            
        data = request.get_json()
        url = data.get('url')
        specialty = data.get('specialty')
        description = data.get('description')
        
        if not url:
            return error_response("URL requerida", 400)
            
        # 2. Validar dominio
        if not WebScraper.is_allowed_domain(url):
            return error_response(
                "Dominio no permitido. Solo sitios oficiales (.gob.mx)", 
                400
            )
            
        # 3. Fetch inmediato del contenido
        scrape_result = WebScraper.fetch_and_clean(url)
        
        if not scrape_result['success']:
            return error_response(scrape_result['error'], 400)
            
        # 4. Guardar en BD
        prof_url = ProfessionalURL(
            professional_id=professional_id,
            url=url,
            specialty=specialty,
            description=description,
            is_active=True,
            last_fetched=datetime.utcnow(),
            cached_content=scrape_result['content']
        )
        
        db.session.add(prof_url)
        db.session.commit()
        
        return success_response(
            {'url': prof_url.to_dict()},
            "URL agregada y procesada correctamente",
            201
        )
        
    except Exception as e:
        db.session.rollback()
        return error_response(f"Error agregando URL: {str(e)}", 500)

@profesionales_bp.route('/<int:professional_id>/urls', methods=['GET'])
@login_required
def get_urls(professional_id):
    """
    GET /api/profesionales/{id}/urls
    Listar URLs del profesional
    """
    try:
        prof = Professional.query.get(professional_id)
        if not prof:
            return error_response("Profesional no encontrado", 404)
            
        # Permitir ver URLs si es el dueño o admin (o incluso público si se requiere)
        # Por ahora restringimos al dueño para edición
        if prof.user_id != request.current_user_id:
            return error_response("No tienes permiso", 403)
            
        urls = ProfessionalURL.query.filter_by(
            professional_id=professional_id,
            is_active=True
        ).order_by(ProfessionalURL.created_at.desc()).all()
        
        return success_response({
            'urls': [u.to_dict() for u in urls],
            'total': len(urls)
        })
        
    except Exception as e:
        return error_response(f"Error listando URLs: {str(e)}", 500)

@profesionales_bp.route('/<int:professional_id>/urls/<int:url_id>', methods=['DELETE'])
@login_required
def delete_url(professional_id, url_id):
    """
    DELETE /api/profesionales/{id}/urls/{url_id}
    Eliminar (desactivar) URL
    """
    try:
        prof = Professional.query.get(professional_id)
        if not prof:
            return error_response("Profesional no encontrado", 404)
            
        if prof.user_id != request.current_user_id:
            return error_response("No tienes permiso", 403)
            
        url_obj = ProfessionalURL.query.get(url_id)
        if not url_obj or url_obj.professional_id != professional_id:
            return error_response("URL no encontrada", 404)
            
        # Hard delete o Soft delete? El usuario pidió "Eliminar de BD" en checklist
        # Haremos hard delete para limpiar
        db.session.delete(url_obj)
        db.session.commit()
        
        return success_response(None, "URL eliminada correctamente")
        
    except Exception as e:
        db.session.rollback()
        return error_response(f"Error eliminando URL: {str(e)}", 500)

@profesionales_bp.route('/<int:professional_id>/urls/<int:url_id>/refresh', methods=['POST'])
@login_required
def refresh_url(professional_id, url_id):
    """
    POST /api/profesionales/{id}/urls/{url_id}/refresh
    Actualizar contenido de una URL
    """
    try:
        prof = Professional.query.get(professional_id)
        if not prof:
            return error_response("Profesional no encontrado", 404)
            
        if prof.user_id != request.current_user_id:
            return error_response("No tienes permiso", 403)
            
        url_obj = ProfessionalURL.query.get(url_id)
        if not url_obj or url_obj.professional_id != professional_id:
            return error_response("URL no encontrada", 404)
            
        # Re-scrape
        scrape_result = WebScraper.fetch_and_clean(url_obj.url)
        
        if not scrape_result['success']:
            return error_response(scrape_result['error'], 400)
            
        # Update DB
        url_obj.cached_content = scrape_result['content']
        url_obj.last_fetched = datetime.utcnow()
        db.session.commit()
        
        return success_response(
            {'url': url_obj.to_dict()},
            "Contenido actualizado correctamente"
        )
        
    except Exception as e:
        db.session.rollback()
        return error_response(f"Error actualizando URL: {str(e)}", 500)
        


