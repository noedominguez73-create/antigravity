from app import create_app, db
from app.models import User, Professional

app = create_app()

with app.app_context():
    email = 'noedominguez73@gmail.com'
    print(f"Inspecting user: {email}")
    
    user = User.query.filter_by(email=email).first()
    if not user:
        print("User not found.")
    else:
        print(f"User ID: {user.id}")
        print(f"User Role: {user.role}")
        
        prof = Professional.query.filter_by(user_id=user.id).first()
        if prof:
            print(f"Professional Profile Found. ID: {prof.id}")
            print(f"Professional Owner User ID: {prof.user_id}")
        else:
            print("No Professional profile found for this user.")
            
        # Check if there are ANY professionals
        all_profs = Professional.query.all()
        print(f"\nTotal Professionals in DB: {len(all_profs)}")
        for p in all_profs:
            print(f"Prof ID: {p.id}, User ID: {p.user_id}, Name: {p.user.full_name}, Email: {p.user.email}")
