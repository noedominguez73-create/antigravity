import sys, os

# Agregar el directorio actual al path
sys.path.append(os.getcwd())

# Importar la aplicación Flask desde run.py
# Passenger busca un objeto llamado 'application' por defecto
from run import app as application
