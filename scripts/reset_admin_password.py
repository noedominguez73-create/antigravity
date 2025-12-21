import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()

def reset_admin():
    with app.app_context():
        # Find admin
        admin = User.query.filter_by(email='admin@asesoriaimss.io').first()
        
        if not admin:
            print("Admin user not found. Creating one...")
            admin = User(
                email='admin@asesoriaimss.io',
                password_hash=generate_password_hash('admin123'),
                role='admin',
                full_name='Administrador'
            )
            db.session.add(admin)
        else:
            print("Admin user found. Resetting password...")
            admin.password_hash = generate_password_hash('admin123')
            
        db.session.commit()
        print("SUCCESS: Password reset to 'admin123'")

if __name__ == '__main__':
    reset_admin()
