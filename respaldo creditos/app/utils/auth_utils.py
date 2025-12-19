# Authentication Utilities - JWT Token Management

import jwt
import os
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from app.models import User

SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError(
        "SECRET_KEY must be set in environment variables. "
        "This is required for secure JWT token generation."
    )
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

def generate_token(user_id, role='user'):
    """Generate JWT token for authenticated user"""
    payload = {
        'user_id': user_id,
        'role': role,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

def decode_token(token):
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_token_from_header():
    """Extract token from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    return parts[1]

def login_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_token_from_header()
        
        if not token:
            return jsonify({'error': 'Token de autenticación requerido'}), 401
        
        payload = decode_token(token)
        if not payload:
            return jsonify({'error': 'Token inválido o expirado'}), 401
        
        # Attach user info to request
        request.current_user_id = payload['user_id']
        request.current_user_role = payload['role']
        
        return f(*args, **kwargs)
    
    return decorated_function

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_token_from_header()
        
        if not token:
            return jsonify({'error': 'Token de autenticación requerido'}), 401
        
        payload = decode_token(token)
        if not payload:
            return jsonify({'error': 'Token inválido o expirado'}), 401
        
        if payload['role'] != 'admin':
            return jsonify({'error': 'Acceso denegado. Se requieren privilegios de administrador'}), 403
        
        request.current_user_id = payload['user_id']
        request.current_user_role = payload['role']
        
        return f(*args, **kwargs)
    
    return decorated_function

def professional_required(f):
    """Decorator to require professional role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_token_from_header()
        
        if not token:
            return jsonify({'error': 'Token de autenticación requerido'}), 401
        
        payload = decode_token(token)
        if not payload:
            return jsonify({'error': 'Token inválido o expirado'}), 401
        
        if payload['role'] not in ['professional', 'admin']:
            return jsonify({'error': 'Acceso denegado. Se requiere cuenta profesional'}), 403
        
        request.current_user_id = payload['user_id']
        request.current_user_role = payload['role']
        
        return f(*args, **kwargs)
    
    return decorated_function
