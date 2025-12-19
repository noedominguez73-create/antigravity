# Comments Routes - Reviews and Ratings

from flask import Blueprint, request, jsonify
from app import db
from app.models import Comment, Professional, User
from app.utils.auth_utils import login_required, admin_required
from app.utils.validators import validate_rating, validate_required_fields, error_response, success_response

comentarios_bp = Blueprint('comentarios', __name__, url_prefix='/api/comentarios')

@comentarios_bp.route('/<int:profesional_id>', methods=['POST'])
@login_required
def create_comment(profesional_id):
    """Create comment for professional"""
    data = request.get_json()
    
    # Validate required fields
    required = ['rating', 'content']
    valid, message = validate_required_fields(data, required)
    if not valid:
        return error_response(message)
    
    # Validate rating
    if not validate_rating(data['rating']):
        return error_response('Calificación debe ser entre 1 y 5')
    
    # Check if professional exists
    prof = Professional.query.get(profesional_id)
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Check if user already commented
    existing = Comment.query.filter_by(
        professional_id=profesional_id,
        user_id=request.current_user_id
    ).first()
    
    if existing:
        return error_response('Ya has comentado sobre este profesional', 409)
    
    # Create comment
    try:
        comment = Comment(
            professional_id=profesional_id,
            user_id=request.current_user_id,
            rating=int(data['rating']),
            content=data['content'],
            status='pending'  # Requires admin approval
        )
        db.session.add(comment)
        db.session.commit()
        
        return success_response({
            'id': comment.id,
            'status': comment.status
        }, 'Comentario enviado. Pendiente de aprobación.', 201)
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al crear comentario: {str(e)}', 500)

@comentarios_bp.route('/<int:profesional_id>', methods=['GET'])
def get_comments(profesional_id):
    """Get approved comments for professional"""
    # Only show approved comments to public
    comments = Comment.query.filter_by(
        professional_id=profesional_id,
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

@comentarios_bp.route('/<int:id>', methods=['PUT'])
@login_required
def update_comment(id):
    """Edit own comment"""
    comment = Comment.query.get(id)
    
    if not comment:
        return error_response('Comentario no encontrado', 404)
    
    # Check ownership
    if comment.user_id != request.current_user_id:
        return error_response('No tienes permiso para editar este comentario', 403)
    
    data = request.get_json()
    
    # Update fields
    if 'rating' in data:
        if not validate_rating(data['rating']):
            return error_response('Calificación debe ser entre 1 y 5')
        comment.rating = int(data['rating'])
    
    if 'content' in data:
        comment.content = data['content']
    
    # Reset to pending after edit
    comment.status = 'pending'
    
    try:
        db.session.commit()
        return success_response(None, 'Comentario actualizado. Pendiente de aprobación.')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al actualizar comentario: {str(e)}', 500)

@comentarios_bp.route('/<int:id>', methods=['DELETE'])
@login_required
def delete_comment(id):
    """Delete own comment"""
    comment = Comment.query.get(id)
    
    if not comment:
        return error_response('Comentario no encontrado', 404)
    
    # Check ownership or admin
    if comment.user_id != request.current_user_id and request.current_user_role != 'admin':
        return error_response('No tienes permiso para eliminar este comentario', 403)
    
    try:
        db.session.delete(comment)
        db.session.commit()
        return success_response(None, 'Comentario eliminado exitosamente')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al eliminar comentario: {str(e)}', 500)

@comentarios_bp.route('/<int:id>/aprobar', methods=['POST'])
@admin_required
def approve_comment(id):
    """Admin: Approve comment"""
    comment = Comment.query.get(id)
    
    if not comment:
        return error_response('Comentario no encontrado', 404)
    
    comment.status = 'approved'
    
    # Update professional rating
    prof = comment.professional
    approved_comments = Comment.query.filter_by(
        professional_id=prof.id,
        status='approved'
    ).all()
    
    if approved_comments:
        total_rating = sum(c.rating for c in approved_comments)
        prof.rating = total_rating / len(approved_comments)
        prof.total_reviews = len(approved_comments)
    
    try:
        db.session.commit()
        return success_response(None, 'Comentario aprobado exitosamente')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al aprobar comentario: {str(e)}', 500)

@comentarios_bp.route('/<int:id>/rechazar', methods=['POST'])
@admin_required
def reject_comment(id):
    """Admin: Reject comment"""
    comment = Comment.query.get(id)
    
    if not comment:
        return error_response('Comentario no encontrado', 404)
    
    comment.status = 'rejected'
    
    try:
        db.session.commit()
        return success_response(None, 'Comentario rechazado')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al rechazar comentario: {str(e)}', 500)
