import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User
from werkzeug.security import check_password_hash, generate_password_hash

app = create_app()

def diagnose():
    with app.app_context():
        print("--- DIAGNOSTICS START ---")
        
        # 1. Find Admin
        admin = User.query.filter_by(role='admin').first()
        if not admin:
            print("ERROR: No admin user found!")
            return

        print(f"Admin ID: {admin.id}")
        print(f"Email: '{admin.email}'")
        print(f"Phone: '{admin.phone_number}'")
        print(f"Hash: {admin.password_hash[:20]}...")
        
        # 2. Check Password
        input_pass = '1239'
        is_valid = check_password_hash(admin.password_hash, input_pass)
        print(f"Password '{input_pass}' Valid? -> {is_valid}")
        
        if not is_valid:
            print("Resetting password to '1239' just in case...")
            admin.password_hash = generate_password_hash(input_pass)
            db.session.commit()
            print("Password check after reset:", check_password_hash(admin.password_hash, input_pass))

        # 3. Test Phone Query
        target_phone = '+526562173335'
        found_by_phone = User.query.filter(User.phone_number == target_phone).first()
        print(f"Query by phone '{target_phone}' found user? -> {found_by_phone is not None}")
        
        if not found_by_phone:
            print(f"WARNING: Could not find user by phone '{target_phone}'.")
            # Check for hidden chars
            print(f"Actual DB Phone Bytes: {admin.phone_number.encode('utf-8') if admin.phone_number else 'None'}")
            print(f"Target Phone Bytes: {target_phone.encode('utf-8')}")

        print("--- DIAGNOSTICS END ---")

if __name__ == '__main__':
    diagnose()
