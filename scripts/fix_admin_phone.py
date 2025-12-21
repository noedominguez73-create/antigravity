import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User

app = create_app()

def fix_admin_phone():
    with app.app_context():
        admin = User.query.filter_by(role='admin').first()
        if admin:
            print(f"Current Phone: '{admin.phone_number}'")
            # Force set to compact format
            new_phone = '+526562173335'
            admin.phone_number = new_phone
            db.session.commit()
            print(f"New Phone set to: '{new_phone}'")
        else:
            print("Admin not found")

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fix_admin_phone()
