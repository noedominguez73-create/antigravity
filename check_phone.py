
from app import create_app
from app.models import Credit, Professional, User

app = create_app()

with app.app_context():
    # Find pending credits
    pending_credits = Credit.query.filter_by(payment_status='pending').all()
    
    if not pending_credits:
        print("No pending credits found.")
    else:
        print(f"Found {len(pending_credits)} pending transactions:")
        for credit in pending_credits:
            professional = Professional.query.get(credit.professional_id)
            user = User.query.get(professional.user_id) if professional else None
            
            p_name = user.full_name if user else "Unknown"
            p_phone = user.phone_number if user else "None"
            p_email = user.email if user else "None"
            
            print(f"Transaction ID: {credit.id}")
            print(f"  User: {p_name}")
            print(f"  Email: {p_email}")
            print(f"  Phone: {p_phone}")
            print(f"  Amount: {credit.transaction_amount}")
            print("-" * 20)
