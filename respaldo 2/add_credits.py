from app import create_app, db
from app.models import Credit

app = create_app()

with app.app_context():
    prof_id = 3
    amount = 100
    
    credit = Credit(
        professional_id=prof_id,
        transaction_type='admin_addition',
        transaction_amount=amount,
        payment_status='confirmed'
    )
    
    db.session.add(credit)
    db.session.commit()
    
    print(f"Successfully added {amount} credits to professional {prof_id}")
