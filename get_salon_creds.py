from app import create_app, db
from app.models import SalonConfig, User

app = create_app()

with app.app_context():
    salons = SalonConfig.query.all()
    if not salons:
        print("No salons found.")
    else:
        for salon in salons:
            user = User.query.get(salon.user_id)
            print(f"Salon: {salon.salon_name}")
            print(f"User Email: {user.email}")
            print(f"User ID: {user.id}")
            print("-" * 20)
