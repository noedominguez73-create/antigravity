from app import create_app, db
from app.models import SalonConfig, User

app = create_app()

with app.app_context():
    import logging
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    salons = SalonConfig.query.all()
    with open('creds.txt', 'w') as f:
        for salon in salons:
            user = User.query.get(salon.user_id)
            f.write(f"EMAIL={user.email}\n")
            f.write(f"PASS=123456\n")
