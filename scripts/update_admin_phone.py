import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()

def update_admin_creds():
    with app.app_context():
        # Find admin (we assume ID 1 or the email we just set)
        admin = User.query.filter_by(role='admin').first()
        
        if not admin:
            print("Admin not found!")
            return

        print(f"Updating Admin: {admin.email}")
        
        # Update Phone
        admin.phone_number = '+52 6562173335'
        
        # Update Password
        admin.password_hash = generate_password_hash('1239')
        
        db.session.commit()
        print("SUCCESS: Credentials updated.")
        print("Phone: +52 6562173335")
        print("Pass: 1239")

if __name__ == '__main__':
    # Fix path if needed, similar to previous scripts
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    update_admin_creds()
