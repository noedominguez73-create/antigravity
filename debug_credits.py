from app import create_app, db
from app.models import Professional, Credit, User
import sys

app = create_app()

with app.app_context():
    with open('debug_output.txt', 'w') as f:
        f.write("--- PROFESSIONALS ---\n")
        profs = Professional.query.all()
        for p in profs:
            f.write(f"ID: {p.id} | Name: {p.user.full_name} | Email: {p.user.email}\n")
            
        f.write("\n--- ALL CREDITS ---\n")
        credits = Credit.query.all()
        if not credits:
            f.write("NO CREDITS FOUND IN DB\n")
        for c in credits:
            f.write(f"CreditID: {c.id} | ProfID: {c.professional_id} | Type: {c.transaction_type} | Amount: {c.transaction_amount} | Status: {c.payment_status}\n")
