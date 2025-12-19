# Credits Routes - Purchase and Management

from flask import Blueprint, request, jsonify
from app import db
from app.models import Credit, Professional
from app.utils.auth_utils import professional_required, admin_required
from app.utils.validators import validate_required_fields, success_response
from app.utils.error_handler import error_response

creditos_bp = Blueprint('creditos', __name__, url_prefix='/api/creditos')

# Credit pricing (placeholder)
CREDIT_PRICE_MXN = 0.30  # $0.30 MXN per credit

@creditos_bp.route('/comprar', methods=['POST'])
@professional_required
def purchase_credits():
    """Purchase credits (PLACEHOLDER payment integration)"""
    data = request.get_json()
    
    # Validate required fields
    required = ['amount', 'payment_method']
    valid, message = validate_required_fields(data, required)
    if not valid:
        return error_response(message)
    
    amount = data['amount']  # Number of credits
    payment_method = data['payment_method']  # 'clabe', 'oxxo', 'efectivo'
    
    if payment_method not in ['clabe', 'oxxo', 'efectivo']:
        return error_response('Método de pago inválido. Opciones: clabe, oxxo, efectivo')
    
    if amount < 1:
        return error_response('La cantidad mínima es 1 crédito')
    
    # Get professional profile
    prof = Professional.query.filter_by(user_id=request.current_user_id).first()
    if not prof:
        return error_response('Perfil profesional no encontrado', 404)
    
    # Calculate price
    total_price = amount * CREDIT_PRICE_MXN
    
    # Create credit transaction (pending payment)
    try:
        credit = Credit(
            professional_id=prof.id,
            transaction_type='purchase',
            transaction_amount=amount,
            payment_method=payment_method,
            payment_status='pending',
            price_mxn=total_price
        )
        db.session.add(credit)
        db.session.commit()
        
        # PLACEHOLDER: In production, generate payment instructions
        payment_instructions = {
            'clabe': 'Transferir a CLABE: 012345678901234567',
            'oxxo': 'Código de pago OXXO: ABC123456',
            'efectivo': 'Contactar al administrador para pago en efectivo'
        }
        
        return success_response({
            'transaction_id': credit.id,
            'amount': amount,
            'total_price_mxn': total_price,
            'payment_method': payment_method,
            'payment_status': 'pending',
            'instructions': payment_instructions.get(payment_method)
        }, 'Transacción creada. Completa el pago para activar los créditos.', 201)
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al crear transacción: {str(e)}', 500)

@creditos_bp.route('/<int:profesional_id>', methods=['GET'])
@professional_required
def get_credits(profesional_id):
    """Check credit balance"""
    # Get professional
    prof = Professional.query.get(profesional_id)
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Check ownership
    if prof.user_id != request.current_user_id and request.current_user_role != 'admin':
        return error_response('No tienes permiso para ver estos créditos', 403)
    
    # Calculate total credits (confirmed purchases)
    total_purchased = db.session.query(db.func.sum(Credit.transaction_amount)).filter(
        Credit.professional_id == profesional_id,
        Credit.transaction_type == 'purchase',
        Credit.payment_status.in_(['confirmed', 'completed'])
    ).scalar() or 0
    
    # Calculate used credits (from chat messages)
    from app.models import ChatMessage
    total_used = db.session.query(db.func.sum(ChatMessage.credits_used)).filter(
        ChatMessage.professional_id == profesional_id
    ).scalar() or 0
    
    # Calculate referral bonuses
    referral_bonuses = db.session.query(db.func.sum(Credit.transaction_amount)).filter(
        Credit.professional_id == profesional_id,
        Credit.transaction_type == 'referral_bonus',
        Credit.payment_status.in_(['confirmed', 'completed'])
    ).scalar() or 0
    
    # Calculate admin additions
    admin_additions = db.session.query(db.func.sum(Credit.transaction_amount)).filter(
        Credit.professional_id == profesional_id,
        Credit.transaction_type == 'admin_addition',
        Credit.payment_status.in_(['confirmed', 'completed'])
    ).scalar() or 0
    
    total_credits = total_purchased + referral_bonuses + admin_additions
    available = total_credits - total_used
    
    # Get recent transactions (purchases/bonuses)
    transactions = Credit.query.filter_by(professional_id=profesional_id).all()
    
    # Get recent usage (chat messages)
    chat_usage = ChatMessage.query.filter_by(
        professional_id=profesional_id
    ).filter(ChatMessage.credits_used > 0).all()
    
    history = []
    
    # Process transactions
    for t in transactions:
        history.append({
            'id': f'tx-{t.id}',
            'type': t.transaction_type,
            'amount': t.transaction_amount,
            'description': f'Transacción: {t.transaction_type}',
            'payment_method': t.payment_method,
            'status': t.payment_status,
            'created_at': t.created_at,
            'is_usage': False
        })
        
    # Process usage
    for c in chat_usage:
        history.append({
            'id': f'msg-{c.id}',
            'type': 'usage',
            'amount': -c.credits_used, # Negative for usage
            'description': f'Uso en chat (Sesión: {c.session_id[:8]}...)',
            'payment_method': 'N/A',
            'status': 'completed',
            'created_at': c.created_at,
            'is_usage': True
        })
        
    # Sort by date descending
    history.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Limit to recent 50
    history = history[:50]
    
    # Format dates for JSON
    for h in history:
        h['created_at'] = h['created_at'].isoformat()
    
    # Check if low on credits
    warning = None
    if total_credits > 0 and available <= total_credits * 0.2:
        warning = f'Advertencia: Te quedan {available} créditos ({int(available/total_credits*100)}%)'
    
    return success_response({
        'total_purchased': total_purchased,
        'referral_bonuses': referral_bonuses,
        'admin_additions': admin_additions,
        'total_credits': total_credits,
        'used': total_used,
        'available': available,
        'warning': warning,
        'recent_transactions': history
    })

@creditos_bp.route('/confirmar-pago', methods=['POST'])
@admin_required
def confirm_payment():
    """Admin: Confirm payment and activate credits"""
    data = request.get_json()
    
    transaction_id = data.get('transaction_id')
    
    if not transaction_id:
        return error_response('transaction_id requerido')
    
    credit = Credit.query.get(transaction_id)
    
    if not credit:
        return error_response('Transacción no encontrada', 404)
    
    if credit.payment_status == 'confirmed':
        return error_response('El pago ya fue confirmado', 409)
    
    credit.payment_status = 'confirmed'
    
    try:
        db.session.commit()
        return success_response({
            'transaction_id': credit.id,
            'professional_id': credit.professional_id,
            'credits_activated': credit.transaction_amount
        }, 'Pago confirmado y créditos activados')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al confirmar pago: {str(e)}', 500)
