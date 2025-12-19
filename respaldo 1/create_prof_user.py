from app import create_app, db
from app.models import User, Professional
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    # Check if user exists
    user = User.query.filter_by(email='prof@test.com').first()
    if not user:
        user = User(
            email='prof@test.com',
            password_hash=generate_password_hash('password123'),
            role='professional',
            full_name='Test Professional'
        )
        db.session.add(user)
        db.session.commit()
        print("User created.")
        
        # Create professional profile
        prof = Professional(
            user_id=user.id,
            bio='Experto en IMSS',
            specialty='Pensiones, Modalidad 40',
            city='CDMX'
        )
        db.session.add(prof)
        db.session.commit()
        print("Professional profile created.")
    else:
        print("User already exists.")
