from app import create_app, db
from app.models import SalonConfig, User
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    # Disable sqlalchemy logging for this
    import logging
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    salons = SalonConfig.query.all()
    if not salons:
        print("RESULT: No salons found.")
    else:
        for salon in salons:
            user = User.query.get(salon.user_id)
            # Reset password to 123456
            user.password_hash = generate_password_hash('123456')
            db.session.commit()
            
            print(f"RESULT: USER_EMAIL={user.email}")
            print(f"RESULT: PASSWORD=123456")
