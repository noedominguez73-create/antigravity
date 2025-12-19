import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    """Configurar logging para la aplicación"""
    
    # Crear carpeta de logs si no existe
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    # Handler para archivo (rotante)
    file_handler = RotatingFileHandler(
        'logs/asesoriaimss.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    
    # Formato detallado
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Nivel de logging
    file_handler.setLevel(logging.DEBUG)
    
    # Agregar al logger de Flask
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.DEBUG)
    
    app.logger.info("=== Logging System Initialized ===")
