from flask import Flask
from app.config.logging_config import setup_logging
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    setup_logging(app)
    CORS(app)
    
    # Rate limiting to prevent brute force attacks
    from app.extensions import limiter
    limiter.init_app(app)
    
    # Force SQLite configuration (temporary fix)
    # Security: SECRET_KEY is mandatory
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    if not app.config['SECRET_KEY']:
        raise ValueError(
            "SECRET_KEY must be set in environment variables. "
            "Generate a secure key with: python -c 'import secrets; print(secrets.token_hex(32))'"
        )
    db_path = os.path.join(app.instance_path, 'asesoriaimss.db').replace('\\', '/')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'  # Force SQLite
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        # Import models to ensure they are registered
        from app import models
        
        from app.routes import main_bp
        from app.routes.auth import auth_bp
        from app.routes.profesionales import profesionales_bp
        from app.routes.chatbot import chatbot_bp
        from app.routes.comentarios import comentarios_bp
        from app.routes.creditos import creditos_bp
        from app.routes.referrals import referrals_bp
        from app.routes.admin import admin_bp
        from app.routes.debug import debug_bp
        from app.routes.api import api_bp
        
        app.register_blueprint(main_bp)
        app.register_blueprint(auth_bp)
        app.register_blueprint(profesionales_bp)
        app.register_blueprint(chatbot_bp)
        app.register_blueprint(comentarios_bp)
        app.register_blueprint(creditos_bp)
        app.register_blueprint(referrals_bp)
        app.register_blueprint(admin_bp)
        app.register_blueprint(debug_bp)
        app.register_blueprint(api_bp)
        
        # Initialize RAG Index
        try:
            from app.services.rag_service import rag_service
            rag_service.reload_index()
        except Exception as e:
            app.logger.warning(f"Failed to initialize RAG index: {e}")
    
    return app





