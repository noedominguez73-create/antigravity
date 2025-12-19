# Referrals Routes - Referral System Management

from flask import Blueprint, request, jsonify
from app import db
from app.models import Referral, Professional, User, ReferralEarning, ReferralWithdrawal, Credit
from app.utils.auth_utils import professional_required
from app.utils.validators import validate_clabe, validate_required_fields, error_response, success_response
from datetime import datetime, timedelta
import secrets
import string

referrals_bp = Blueprint('referrals', __name__, url_prefix='/api/referrals')

# Referral configuration
COMMISSION_RATE = 0.20  # 20%
REFERRAL_DURATION_MONTHS = 12
MIN_WITHDRAWAL_MXN = 100.0

def generate_referral_code():
    """Generate unique referral code"""
    chars = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(secrets.choice(chars) for _ in range(8))
        existing = Referral.query.filter_by(referral_code=code).first()
        if not existing:
            return code

@referrals_bp.route('/generar-link', methods=['POST'])
@professional_required
def generate_link():
    """Generate referral link for professional"""
    # Get professional profile
    prof = Professional.query.filter_by(user_id=request.current_user_id).first()
    if not prof:
        return error_response('Perfil profesional no encontrado', 404)
    
    # Generate unique code
    code = generate_referral_code()
    
    # Create referral
    try:
        referral = Referral(
            referrer_id=prof.id,
            referral_code=code,
            commission_rate=COMMISSION_RATE,
            expires_at=datetime.utcnow() + timedelta(days=REFERRAL_DURATION_MONTHS * 30)
        )
        db.session.add(referral)
        db.session.commit()
        
        # Generate referral link (placeholder URL)
        referral_link = f"https://asesoriaimss.io/registro?ref={code}"
        
        return success_response({
            'referral_code': code,
            'referral_link': referral_link,
            'commission_rate': COMMISSION_RATE * 100,  # As percentage
            'expires_at': referral.expires_at.isoformat()
        }, 'Link de referido generado exitosamente', 201)
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al generar link: {str(e)}', 500)

@referrals_bp.route('/registrar-nuevo', methods=['POST'])
def register_new():
    """Register new user via referral code"""
    data = request.get_json()
    
    referral_code = data.get('referral_code')
    
    if not referral_code:
        return error_response('Código de referido requerido')
    
    # Find referral
    referral = Referral.query.filter_by(referral_code=referral_code).first()
    
    if not referral:
        return error_response('Código de referido inválido', 404)
    
    # Check if expired
    if referral.expires_at < datetime.utcnow():
        return error_response('El código de referido ha expirado', 410)
    
    # This endpoint returns referral info for the registration process
    # The actual user registration happens in /api/auth/registro
    # and should link the new user to this referral
    
    return success_response({
        'referral_code': referral_code,
        'referrer_id': referral.referrer_id,
        'commission_rate': referral.commission_rate,
        'valid': True
    }, 'Código de referido válido')

@referrals_bp.route('/<int:profesional_id>/activos', methods=['GET'])
@professional_required
def get_active_referrals(profesional_id):
    """Get active referrals for professional"""
    # Get professional
    prof = Professional.query.get(profesional_id)
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Check ownership
    if prof.user_id != request.current_user_id and request.current_user_role != 'admin':
        return error_response('No tienes permiso para ver estos referidos', 403)
    
    # Get active referrals
    referrals = Referral.query.filter_by(
        referrer_id=profesional_id,
        status='active'
    ).filter(Referral.expires_at > datetime.utcnow()).all()
    
    referrals_list = []
    for ref in referrals:
        # Count referred users
        referred_user = User.query.get(ref.referred_user_id) if ref.referred_user_id else None
        
        referrals_list.append({
            'id': ref.id,
            'referral_code': ref.referral_code,
            'referred_user': referred_user.full_name if referred_user else None,
            'total_earned_mxn': ref.total_earned_mxn,
            'commission_rate': ref.commission_rate * 100,
            'expires_at': ref.expires_at.isoformat(),
            'created_at': ref.created_at.isoformat()
        })
    
    return success_response({'referrals': referrals_list})

@referrals_bp.route('/<int:profesional_id>/ganancias', methods=['GET'])
@professional_required
def get_earnings(profesional_id):
    """Get referral earnings summary"""
    # Get professional
    prof = Professional.query.get(profesional_id)
    if not prof:
        return error_response('Profesional no encontrado', 404)
    
    # Check ownership
    if prof.user_id != request.current_user_id and request.current_user_role != 'admin':
        return error_response('No tienes permiso para ver estas ganancias', 403)
    
    # Calculate total earnings
    total_earned = db.session.query(db.func.sum(Referral.total_earned_mxn)).filter(
        Referral.referrer_id == profesional_id
    ).scalar() or 0.0
    
    # Calculate withdrawn amount
    total_withdrawn = db.session.query(db.func.sum(ReferralWithdrawal.amount_mxn)).filter(
        ReferralWithdrawal.referrer_id == profesional_id,
        ReferralWithdrawal.status.in_(['approved', 'completed'])
    ).scalar() or 0.0
    
    # Calculate pending withdrawals
    pending_withdrawals = db.session.query(db.func.sum(ReferralWithdrawal.amount_mxn)).filter(
        ReferralWithdrawal.referrer_id == profesional_id,
        ReferralWithdrawal.status == 'pending'
    ).scalar() or 0.0
    
    available = total_earned - total_withdrawn - pending_withdrawals
    
    # Get recent earnings
    earnings = db.session.query(ReferralEarning).join(Referral).filter(
        Referral.referrer_id == profesional_id
    ).order_by(ReferralEarning.created_at.desc()).limit(10).all()
    
    earnings_list = []
    for earning in earnings:
        earnings_list.append({
            'id': earning.id,
            'amount_mxn': earning.amount_mxn,
            'created_at': earning.created_at.isoformat()
        })
    
    return success_response({
        'total_earned_mxn': total_earned,
        'total_withdrawn_mxn': total_withdrawn,
        'pending_withdrawals_mxn': pending_withdrawals,
        'available_mxn': available,
        'can_withdraw': available >= MIN_WITHDRAWAL_MXN,
        'min_withdrawal_mxn': MIN_WITHDRAWAL_MXN,
        'recent_earnings': earnings_list
    })

@referrals_bp.route('/solicitar-retiro', methods=['POST'])
@professional_required
def request_withdrawal():
    """Request withdrawal of referral earnings"""
    data = request.get_json()
    
    # Validate required fields
    required = ['amount_mxn', 'withdrawal_method']
    valid, message = validate_required_fields(data, required)
    if not valid:
        return error_response(message)
    
    amount = float(data['amount_mxn'])
    method = data['withdrawal_method']  # 'clabe', 'oxxo', 'credits'
    
    if method not in ['clabe', 'oxxo', 'credits']:
        return error_response('Método de retiro inválido. Opciones: clabe, oxxo, credits')
    
    # Validate CLABE if method is clabe
    if method == 'clabe':
        clabe = data.get('clabe_account')
        if not validate_clabe(clabe):
            return error_response('CLABE inválida. Debe tener 18 dígitos')
    
    # Get professional
    prof = Professional.query.filter_by(user_id=request.current_user_id).first()
    if not prof:
        return error_response('Perfil profesional no encontrado', 404)
    
    # Check minimum withdrawal
    if amount < MIN_WITHDRAWAL_MXN:
        return error_response(f'El retiro mínimo es ${MIN_WITHDRAWAL_MXN} MXN')
    
    # Check available balance
    total_earned = db.session.query(db.func.sum(Referral.total_earned_mxn)).filter(
        Referral.referrer_id == prof.id
    ).scalar() or 0.0
    
    total_withdrawn = db.session.query(db.func.sum(ReferralWithdrawal.amount_mxn)).filter(
        ReferralWithdrawal.referrer_id == prof.id,
        ReferralWithdrawal.status.in_(['approved', 'completed'])
    ).scalar() or 0.0
    
    pending = db.session.query(db.func.sum(ReferralWithdrawal.amount_mxn)).filter(
        ReferralWithdrawal.referrer_id == prof.id,
        ReferralWithdrawal.status == 'pending'
    ).scalar() or 0.0
    
    available = total_earned - total_withdrawn - pending
    
    if amount > available:
        return error_response(f'Saldo insuficiente. Disponible: ${available:.2f} MXN')
    
    # Create withdrawal request
    try:
        withdrawal = ReferralWithdrawal(
            referrer_id=prof.id,
            amount_mxn=amount,
            withdrawal_method=method,
            clabe_account=data.get('clabe_account') if method == 'clabe' else None,
            status='pending'
        )
        db.session.add(withdrawal)
        db.session.commit()
        
        return success_response({
            'withdrawal_id': withdrawal.id,
            'amount_mxn': amount,
            'method': method,
            'status': 'pending'
        }, 'Solicitud de retiro enviada. Será procesada por el administrador.', 201)
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al solicitar retiro: {str(e)}', 500)
