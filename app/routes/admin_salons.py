
from flask import request, jsonify
from app import db
from app.models import User, SalonConfig
from app.utils.validators import success_response, error_response, validate_required_fields
from app.utils.auth_utils import admin_required
from app.routes.admin import admin_bp
from werkzeug.security import generate_password_hash
from datetime import datetime

@admin_bp.route('/salones', methods=['GET'])
@admin_required
def get_salons():
    """Get all registered salons"""
    # Join with User to get client name
    salons = db.session.query(SalonConfig, User).join(User, SalonConfig.user_id == User.id).all()
    
    results = []
    for salon, user in salons:
        salon_data = {
            'id': salon.id,
            'client_name': user.full_name,
            'client_email': user.email,
            'salon_name': salon.salon_name or 'Sin nombre',
            'address': salon.address or 'Sin dirección',
            'city': salon.city or '',
            'state': salon.state or '',
            'country': salon.country or 'México',
            'start_date': salon.start_date.isoformat() if salon.start_date else None,
            'payment_date': salon.payment_date.isoformat() if salon.payment_date else None,
            'tokens_consumed': salon.tokens_consumed,
            'created_at': salon.created_at.isoformat()
        }
        
        # Calculate Status
        if salon.payment_date:
            today = datetime.now().date()
            delta = (salon.payment_date - today).days
            
            salon_data['days_remaining'] = delta
            
            if delta < 0:
                salon_data['status'] = 'overdue' # Vencido
                salon_data['status_label'] = f'Vencido hace {abs(delta)} días'
                salon_data['status_color'] = 'red'
                salon_data['sort_order'] = 0 # High priority
            elif delta <= 5:
                salon_data['status'] = 'due_soon' # Próximo
                salon_data['status_label'] = f'Vence en {delta} días'
                salon_data['status_color'] = 'amber' # Yellow/Orange
                salon_data['sort_order'] = 1
            else:
                salon_data['status'] = 'active' # Activo
                salon_data['status_label'] = 'Activo'
                salon_data['status_color'] = 'emerald' # Green
                salon_data['sort_order'] = 2
        else:
            salon_data['status'] = 'unknown'
            salon_data['status_label'] = 'Sin fecha'
            salon_data['status_color'] = 'gray'
            salon_data['sort_order'] = 3
            
        results.append(salon_data)
        
    # Sort by priority (Overdue -> Due Soon -> Active)
    results.sort(key=lambda x: x['sort_order'])
    
    # Clean up sort key before sending if desired, but keeping it is fine
        
    return success_response({'salons': results})

@admin_bp.route('/salones', methods=['POST'])
@admin_required
def create_salon():
    """Create a new salon manually"""
    data = request.get_json()
    
    # Validation
    # 'client_identifier' replaces 'client_email'
    required = ['client_identifier', 'salon_name', 'address', 'city', 'state', 'country', 'start_date', 'payment_date']
    valid, message = validate_required_fields(data, required)
    if not valid:
        return error_response(message)
    
    # Find user/client by Email OR Phone
    identifier = data['client_identifier']
    user = User.query.filter((User.email == identifier) | (User.phone_number == identifier)).first()
    
    # Auto-create user if not exists
    if not user:
        client_name = data.get('client_name')
        if not client_name:
             return error_response('El usuario no existe. Debe proporcionar el "Nombre del Cliente" para crearlo.', 400)
             
        try:
            # Determine if identifier is email or phone
            new_email = identifier if '@' in identifier else None
            new_phone = identifier if not new_email else None
            
            # Placeholder email if only phone is provided (required by some schemas?)
            # Assuming User model allows nullable email or we generate a dummy one
            if not new_email:
                import uuid
                new_email = f"user_{uuid.uuid4().hex[:8]}@completmirror.io" # Temp email
            
            user = User(
                email=new_email,
                full_name=client_name,
                role='user', # Or 'salon_owner'
                phone_number=new_phone
            )
            user.password_hash = generate_password_hash('Temporal123!') # Default password
            db.session.add(user)
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            return error_response(f'Error al crear usuario automático: {str(e)}', 500)
    
    # Check if user already has a salon
    if SalonConfig.query.filter_by(user_id=user.id).first():
        return error_response('Este usuario ya tiene un salón registrado.', 400)
    
    try:
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
        payment_date = datetime.strptime(data['payment_date'], '%Y-%m-%d').date()
    except ValueError:
        return error_response('Formato de fecha inválido. Use YYYY-MM-DD')
        
    try:
        new_salon = SalonConfig(
            user_id=user.id,
            salon_name=data['salon_name'],
            address=data['address'],
            city=data['city'],
            state=data['state'],
            country=data['country'],
            start_date=start_date,
            payment_date=payment_date,
            tokens_consumed=int(data.get('tokens_consumed', 0)),
            # Defaults
            primary_color='#000000',
            secondary_color='#ffffff',
            stylist_name='Asistente'
        )
        
        db.session.add(new_salon)
        db.session.commit()
        
        return success_response({
            'id': new_salon.id, 
            'salon_name': new_salon.salon_name
        }, 'Salón creado exitosamente')
        
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error al crear salón: {str(e)}', 500)

@admin_bp.route('/salones/<int:id>', methods=['DELETE'])
@admin_required
def delete_salon(id):
    """Delete a salon"""
    salon = SalonConfig.query.get(id)
    if not salon:
        return error_response('Salón no encontrado', 404)
        
    try:
        db.session.delete(salon)
        db.session.commit()
        return success_response(None, 'Salón eliminado')
    except Exception as e:
        db.session.rollback()
        return error_response(f'Error: {str(e)}', 500)
