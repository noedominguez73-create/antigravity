from app import create_app, db
from app.models import Professional, Credit, User
from app.routes.admin import add_credits_manually
from flask import Flask, request

app = create_app()

with app.app_context():
    # Mock request context
    with app.test_request_context(
        '/api/admin/creditos/4',
        method='POST',
        json={'amount': 100, 'reason': 'Test script addition'},
        headers={'Authorization': 'Bearer ...'} # We bypass auth decorator for unit test if possible, but here we are calling the function directly? 
        # No, we can't call route function directly easily because of decorators.
        # Better to use test client.
    ):
        pass

    # Let's use the test client
    client = app.test_client()
    
    # We need to login as admin first to get a token
    # Assuming we can generate a token or bypass.
    # Let's just manually insert the record using the SAME logic as admin.py to see if it commits.
    
    print("Attempting to add credits manually via DB session...")
    try:
        new_credit = Credit(
            professional_id=4,
            amount=100,
            transaction_type='admin_addition',
            transaction_amount=100,
            payment_method='admin',
            payment_status='confirmed',
            price_mxn=0.0
        )
        db.session.add(new_credit)
        db.session.commit()
        print("Successfully added credit record.")
    except Exception as e:
        print(f"Error adding credit: {e}")
        
    # Verify
    c = Credit.query.filter_by(professional_id=4).first()
    if c:
        print(f"Found credit: ID {c.id}, Status {c.payment_status}, Amount {c.transaction_amount}")
    else:
        print("Credit NOT found after commit!")
