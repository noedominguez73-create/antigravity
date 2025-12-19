from flask import request
from app import db
from app.models import Credit, Professional, User
from app.utils.validators import success_response, error_response
from app.utils.auth_utils import admin_required
from app.routes.admin import admin_bp


@admin_bp.route('/transacciones', methods=['GET'])
@admin_required
def get_pending_transactions():
    """Get all pending credit transactions"""
    try:
        # Get pending credit purchases
        pending_credits = Credit.query.filter_by(
            payment_status='pending'
        ).order_by(Credit.created_at.desc()).all()
        
        transactions = []
        for credit in pending_credits:
            professional = Professional.query.get(credit.professional_id)
            user = User.query.get(professional.user_id) if professional else None
            
            transactions.append({
                'id': credit.id,
                'professional_id': credit.professional_id,
                'professional_name': user.full_name if user else 'Desconocido',
                'amount': credit.transaction_amount,
                'price_mxn': credit.price_mxn,
                'payment_method': credit.payment_method,
                'status': credit.payment_status,
                'created_at': credit.created_at.isoformat() if credit.created_at else None
            })
        
        return success_response({
            'transactions': transactions,
            'total': len(transactions)
        })
        
    except Exception as e:
        return error_response(f'Error al obtener transacciones: {str(e)}', 500)


@admin_bp.route('/transacciones/<int:transaction_id>/aprobar', methods=['POST'])
@admin_required
def approve_transaction(transaction_id):
    """Approve a pending credit transaction"""
    try:
        credit = Credit.query.get(transaction_id)
        
        if not credit:
            return error_response('Transacción no encontrada', 404)
        
        if credit.payment_status != 'pending':
            return error_response('La transacción ya fue procesada')
        
        # Update transaction status
        credit.payment_status = 'completed'
        db.session.commit()
        
        return success_response({
            'transaction_id': transaction_id,
            'status': 'completed'
        }, 'Transacción aprobada exitosamente')
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al aprobar transacción: {str(e)}', 500)


@admin_bp.route('/transacciones/<int:transaction_id>/rechazar', methods=['POST'])
@admin_required
def reject_transaction(transaction_id):
    """Reject a pending credit transaction"""
    try:
        credit = Credit.query.get(transaction_id)
        
        if not credit:
            return error_response('Transacción no encontrada', 404)
        
        if credit.payment_status != 'pending':
            return error_response('La transacción ya fue procesada')
        
        # Update transaction status
        credit.payment_status = 'rejected'
        db.session.commit()
        
        return success_response({
            'transaction_id': transaction_id,
            'status': 'rejected'
        }, 'Transacción rechazada')
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al rechazar transacción: {str(e)}', 500)
