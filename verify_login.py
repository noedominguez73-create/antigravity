from app import create_app, db
from app.models import User
from werkzeug.security import check_password_hash, generate_password_hash

app = create_app()

with app.app_context():
    # Disable sqlalchemy logging for this
    import logging
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    email = "user_e446e3ee@completmirror.io"
    user = User.query.filter_by(email=email).first()
    
    if not user:
        print(f"ERROR: User with email '{email}' not found!")
    else:
        print(f"FOUND: User ID {user.id}")
        print(f"Stored Hash: {user.password_hash[:20]}...")
        
        is_valid = check_password_hash(user.password_hash, '123456')
        print(f"Check '123456': {is_valid}")
        
        if not is_valid:
            print("Resetting password again...")
            new_hash = generate_password_hash('123456')
            user.password_hash = new_hash
            db.session.commit()
            print("Password reset committed.")
            
            # Verify immediately
            user_verify = User.query.filter_by(email=email).first()
            print(f"Re-Check '123456': {check_password_hash(user_verify.password_hash, '123456')}")
