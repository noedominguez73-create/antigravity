# Input Validation Utilities

import re
from flask import jsonify

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength (min 8 chars, 1 uppercase, 1 number)"""
    if len(password) < 8:
        return False, "La contraseña debe tener al menos 8 caracteres"
    
    if not re.search(r'[A-Z]', password):
        return False, "La contraseña debe contener al menos una letra mayúscula"
    
    if not re.search(r'\d', password):
        return False, "La contraseña debe contener al menos un número"
    
    return True, "Contraseña válida"

def validate_rating(rating):
    """Validate rating is between 1 and 5"""
    try:
        rating = int(rating)
        return 1 <= rating <= 5
    except (ValueError, TypeError):
        return False

def validate_clabe(clabe):
    """Validate Mexican CLABE account (18 digits)"""
    if not clabe:
        return False
    
    clabe = str(clabe).strip()
    return len(clabe) == 18 and clabe.isdigit()

def validate_required_fields(data, required_fields):
    """Validate that all required fields are present in data"""
    missing_fields = []
    
    for field in required_fields:
        if field not in data or not data[field]:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Campos requeridos faltantes: {', '.join(missing_fields)}"
    
    return True, "Todos los campos requeridos están presentes"

def error_response(message, status_code=400):
    """Generate standardized error response"""
    return jsonify({'error': message}), status_code

def success_response(data, message=None, status_code=200):
    """Generate standardized success response"""
    response = {'success': True}
    
    if message:
        response['message'] = message
    
    if data:
        response['data'] = data
    
    return jsonify(response), status_code
