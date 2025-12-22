from app import create_app, db
from app.models import User

app = create_app()

with app.app_context():
    # Find the complex user
    complex_email = "user_e446e3ee@completmirror.io"
    user = User.query.filter_by(email=complex_email).first()
    
    if user:
        ns_email = "salon@admin.com"
        user.email = ns_email
        db.session.commit()
        print(f"UPDATED: {complex_email} -> {ns_email}")
    else:
        # Check if already updated?
        user2 = User.query.filter_by(email="salon@admin.com").first()
        if user2:
            print("ALREADY UPDATED: salon@admin.com exists")
        else:
            print("ERROR: User not found!")
