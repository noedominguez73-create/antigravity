from flask import Blueprint, jsonify
from app import db
from app.models import StoreType
from app.utils.validators import success_response

store_bp = Blueprint('store', __name__, url_prefix='/api/tienda')

@store_bp.route('/types', methods=['GET'])
def get_store_types():
    """Get list of available store types, seeding if empty (Lazy Seeding)"""
    try:
        count = StoreType.query.count()
        if count == 0:
            # Seed default types
            default_types = [
                "Estetica",
                "Tienda de ropa",
                "Ferreteria",
                "Artículos varios",
                "Tienda de pisos",
                "Tienda de lamparas"
            ]
            for name in default_types:
                db.session.add(StoreType(name=name))
            db.session.commit()
            
        types = StoreType.query.order_by(StoreType.name.asc()).all()
        return success_response({
            'types': [{'id': t.id, 'name': t.name} for t in types]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
