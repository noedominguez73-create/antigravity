"""
Simple test to check database connection and login
"""
from app import create_app, db
from app.models import User
from werkzeug.security import check_password_hash

app = create_app()

with app.app_context():
    print("=" * 60)
    print("Database Configuration Test")
    print("=" * 60)
    print(f"DATABASE_URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print()
    
    try:
        # Try to query users
        print("Attempting to query users...")
        users = User.query.all()
        print(f"[OK] Successfully connected to database!")
        print(f"[OK] Found {len(users)} users in database")
        print()
        
        if users:
            print("Users in database:")
            for user in users:
                print(f"  - {user.email} (ID: {user.id}, Role: {user.role})")
            print()
            
            # Test login with first user
            test_email = 'juan.perez@example.com'
            test_password = 'password123'
            
            print(f"Testing login with: {test_email}")
            user = User.query.filter_by(email=test_email).first()
            
            if user:
                print(f"[OK] User found: {user.full_name}")
                if check_password_hash(user.password_hash, test_password):
                    print(f"[OK] Password verification successful!")
                else:
                    print(f"[ERROR] Password verification failed!")
            else:
                print(f"[ERROR] User not found!")
                
    except Exception as e:
        print(f"[ERROR] Database error: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
