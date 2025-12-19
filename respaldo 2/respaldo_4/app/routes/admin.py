# Admin Routes - Dashboard and Management

from flask import Blueprint, request, jsonify, current_app
from app import db
from app.models import User, Professional, Comment, Credit, ReferralWithdrawal
from app.utils.auth_utils import admin_required, generate_token
from app.utils.validators import validate_email, validate_required_fields, error_response, success_response
from werkzeug.security import check_password_hash
from datetime import datetime, timedelta

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

@admin_bp.route('/login', methods=['POST'])
def admin_login():
    """Admin authentication with rate limiting (5 attempts per minute)"""
    # Apply rate limiting
    @current_app.limiter.limit("5 per minute")
    def rate_limited_admin_login():
        data = request.get_json()
        
        required = ['email', 'password']
        valid, message = validate_required_fields(data, required)
        if not valid:
            return error_response(message)
        
        # Find admin user
        user = User.query.filter_by(email=data['email'], role='admin').first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return error_response('Credenciales de administrador incorrectas', 401)
        
        # Generate JWT token
        token = generate_token(user.id, 'admin')
        
        return success_response({
            'user': {
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name,
                'role': 'admin'
            },
            'token': token
        }, 'Inicio de sesión de administrador exitoso')

    return rate_limited_admin_login()

@admin_bp.route('/dashboard', methods=['GET'])
@admin_required
def dashboard():
    """Get admin dashboard statistics"""
    # Count users
    total_users = User.query.filter_by(role='user').count()
    total_professionals = Professional.query.filter_by(is_active=True).count()
    
    # Count pending comments
    pending_comments = Comment.query.filter_by(status='pending').count()
    
    # Count pending payments
    pending_payments = Credit.query.filter_by(payment_status='pending').count()
    
    # Count pending withdrawals
    pending_withdrawals = ReferralWithdrawal.query.filter_by(status='pending').count()
    
    # Recent registrations (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_users = User.query.filter(User.created_at >= week_ago).count()
    
    # Total revenue (confirmed credits)
    total_revenue = db.session.query(db.func.sum(Credit.price_mxn)).filter(
        Credit.payment_status == 'confirmed'
    ).scalar() or 0.0
    
    return success_response({
        'users': {
            'total': total_users,
            'recent_week': recent_users
        },
        'professionals': {
            'total': total_professionals
        },
        'pending': {
            'comments': pending_comments,
            'payments': pending_payments,
            'withdrawals': pending_withdrawals
        },
        'revenue': {
            'total_mxn': total_revenue
        }
    })



@admin_bp.route('/comentarios-pendientes', methods=['GET'])
@admin_required
def get_pending_comments():
    """Get all pending comments for review"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    pagination = Comment.query.filter_by(status='pending').order_by(
        Comment.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    comments = []
    for comment in pagination.items:
        comments.append({
            'id': comment.id,
            'professional_id': comment.professional_id,
            'professional_name': comment.professional.user.full_name,
            'author_id': comment.user_id,
            'author_name': comment.author.full_name,
            'rating': comment.rating,
            'content': comment.content,
            'created_at': comment.created_at.isoformat()
        })
    
    return success_response({
        'comments': comments,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })

@admin_bp.route('/comentarios/<int:id>/estado', methods=['PUT'])
@admin_required
def update_comment_status(id):
    """Update comment status (approve/reject)"""
    data = request.get_json()
    
    status = data.get('status')
    
    if status not in ['approved', 'rejected']:
        return error_response('Estado inválido. Opciones: approved, rejected')
    
    comment = Comment.query.get(id)
    
    if not comment:
        return error_response('Comentario no encontrado', 404)
    
    comment.status = status
    
    # If approved, update professional rating
    if status == 'approved':
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
        return success_response(None, f'Comentario {status}')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al actualizar estado: {str(e)}', 500)

@admin_bp.route('/email-marketing', methods=['POST'])
@admin_required
def send_email_marketing():
    """Send marketing email to users (PLACEHOLDER)"""
    data = request.get_json()
    
    required = ['subject', 'content', 'target']
    valid, message = validate_required_fields(data, required)
    if not valid:
        return error_response(message)
    
    subject = data['subject']
    content = data['content']
    target = data['target']  # 'all', 'users', 'professionals'
    
    # Get target users
    query = User.query
    
    if target == 'users':
        query = query.filter_by(role='user')
    elif target == 'professionals':
        query = query.filter_by(role='professional')
    
    users = query.all()
    
    # PLACEHOLDER: In production, integrate with email service (SendGrid, Mailgun, etc.)
    email_count = len(users)
    
    # Log the email campaign (could create EmailCampaign model)
    
    return success_response({
        'subject': subject,
        'target': target,
        'recipients_count': email_count,
        'status': 'queued'
    }, f'Campaña de email enviada a {email_count} usuarios (PLACEHOLDER)')

@admin_bp.route('/pagos-pendientes', methods=['GET'])
@admin_required
def get_pending_payments():
    """Get pending credit payments"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    pagination = Credit.query.filter_by(payment_status='pending').order_by(
        Credit.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    payments = []
    for payment in pagination.items:
        prof = Professional.query.get(payment.professional_id)
        payments.append({
            'id': payment.id,
            'professional_id': payment.professional_id,
            'professional_name': prof.user.full_name if prof else 'N/A',
            'amount_credits': payment.transaction_amount,
            'price_mxn': payment.price_mxn,
            'payment_method': payment.payment_method,
            'created_at': payment.created_at.isoformat()
        })
    
    return success_response({
        'payments': payments,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })

@admin_bp.route('/retiros-pendientes', methods=['GET'])
@admin_required
def get_pending_withdrawals():
    """Get pending referral withdrawals"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    pagination = ReferralWithdrawal.query.filter_by(status='pending').order_by(
        ReferralWithdrawal.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    withdrawals = []
    for withdrawal in pagination.items:
        prof = Professional.query.get(withdrawal.professional_id)
        withdrawals.append({
            'id': withdrawal.id,
            'professional_id': withdrawal.professional_id,
            'professional_name': prof.user.full_name if prof else 'N/A',
            'amount': withdrawal.amount,
            'bank_info': withdrawal.bank_info,
            'created_at': withdrawal.created_at.isoformat()
        })
    
    return success_response({
        'withdrawals': withdrawals,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })

@admin_bp.route('/retiros/<int:id>/estado', methods=['PUT'])
@admin_required
def update_withdrawal_status(id):
    """Update withdrawal status (approve/reject)"""
    data = request.get_json()
    
    status = data.get('status')
    notes = data.get('notes', '')
    
    if status not in ['approved', 'rejected']:
        return error_response('Estado inválido. Opciones: approved, rejected')
    
    withdrawal = ReferralWithdrawal.query.get(id)
    
    if not withdrawal:
        return error_response('Retiro no encontrado', 404)
    
    if withdrawal.status != 'pending':
        return error_response('El retiro ya fue procesado', 400)
    
    withdrawal.status = status
    withdrawal.processed_at = datetime.utcnow()
    withdrawal.notes = notes
    
    # If rejected, refund credits/balance (logic depends on how balance is tracked)
    # Assuming balance is just a calculation, if rejected, it becomes available again
    # No extra action needed if balance is calculated from (earned - withdrawn_approved)
    
    try:
        db.session.commit()
        return success_response(None, f'Retiro {status}')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al actualizar retiro: {str(e)}', 500)

@admin_bp.route('/creditos/<int:professional_id>', methods=['POST'])
@admin_required
def add_credits_manually(professional_id):
    """Manually add credits to a professional"""
    data = request.get_json()
    
    # Validate required fields
    required = ['amount', 'reason']
    valid, message = validate_required_fields(data, required)
    if not valid:
        return error_response(message)
    
    amount = data.get('amount')
    reason = data.get('reason', 'Admin manual addition')
    
    # Validate amount
    try:
        amount = int(amount)
        if amount <= 0:
            return error_response('La cantidad debe ser mayor a 0')
    except (ValueError, TypeError):
        return error_response('Cantidad inválida')
    
    # Check if professional exists
    professional = Professional.query.get(professional_id)
    if not professional:
        return error_response('Profesional no encontrado', 404)
    
    try:
        # Add new credit transaction
        new_credit = Credit(
            professional_id=professional_id,
            amount=amount,
            transaction_type='admin_addition',
            transaction_amount=amount,
            payment_method='admin',
            payment_status='confirmed',  # Changed from 'completed' to 'confirmed' to match creditos.py logic
            price_mxn=0.0
        )
        db.session.add(new_credit)
        db.session.commit()
        
        return success_response({
            'professional_id': professional_id,
            'amount_added': amount,
            'reason': reason
        }, f'{amount} créditos agregados exitosamente')
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al agregar créditos: {str(e)}', 500)

@admin_bp.route('/profesionales/buscar', methods=['GET'])
@admin_required
def search_professionals():
    """Search professionals by name or email"""
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return success_response({'professionals': []})
        
    # Search in User model (name or email) joined with Professional
    # We need professionals, so we query Professional joined with User
    professionals = Professional.query.join(User).filter(
        Professional.is_active == True,
        db.or_(
            User.full_name.ilike(f'%{query}%'),
            User.email.ilike(f'%{query}%')
        )
    ).limit(10).all()
    
    results = []
    for prof in professionals:
        results.append({
            'id': prof.id,
            'full_name': prof.user.full_name,
            'email': prof.user.email,
            'specialty': prof.specialty
        })
        
    return success_response({'professionals': results})

# Import additional admin routes
from app.routes import admin_credits
