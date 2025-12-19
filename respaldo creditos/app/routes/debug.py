# Debug Routes for Testing

from flask import Blueprint, jsonify
from app import db
from app.models import User
from werkzeug.security import check_password_hash

debug_bp = Blueprint('debug', __name__, url_prefix='/api/debug')

@debug_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running'
    })

@debug_bp.route('/config', methods=['GET'])
def config():
    """Show current configuration"""
    from flask import current_app
    return jsonify({
        'database_url': current_app.config.get('SQLALCHEMY_DATABASE_URI'),
        'secret_key_length': len(current_app.config.get('SECRET_KEY', '')),
        'debug': current_app.config.get('DEBUG', False)
    })

@debug_bp.route('/db-test', methods=['GET'])
def db_test():
    """Test database connection"""
    try:
        users = User.query.all()
        return jsonify({
            'status': 'ok',
            'message': 'Database connection successful',
            'user_count': len(users),
            'users': [{'id': u.id, 'email': u.email, 'role': u.role} for u in users]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500

@debug_bp.route('/test-login', methods=['GET'])
def test_login():
    """Test login functionality"""
    try:
        email = 'juan.perez@example.com'
        password = 'password123'
        
        user = User.query.filter_by(email=email).first()
        
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
        
        password_valid = check_password_hash(user.password_hash, password)
        
        return jsonify({
            'status': 'ok',
            'user_found': True,
            'password_valid': password_valid,
            'user': {
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name,
                'role': user.role
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500
