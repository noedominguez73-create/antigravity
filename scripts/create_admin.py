from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash
import sys
import os

# Add project root to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

app = create_app()

def create_admin():
    with app.app_context():
        print("--- CREAR ADMINISTRADOR ---")
        email = input("Email: ").strip()
        password = input("Password: ").strip()
        name = input("Nombre Completo: ").strip()
        
        if not email or not password:
            print("Error: Email y Password requeridos.")
            return

        # Check existing
        existing = User.query.filter_by(email=email).first()
        if existing:
            print(f"Usuario {email} ya existe. Actualizando a ADMIN...")
            existing.role = 'admin'
            existing.password_hash = generate_password_hash(password)
        else:
            print("Creando nuevo usuario Admin...")
            new_admin = User(
                email=email,
                password_hash=generate_password_hash(password),
                full_name=name,
                role='admin',
                is_active=True
            )
            db.session.add(new_admin)
        
        db.session.commit()
        print(f"¡Éxito! Ahora puedes entrar en /admin/login con {email}")

if __name__ == "__main__":
    create_admin()
