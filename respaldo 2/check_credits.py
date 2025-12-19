from app import create_app, db
from app.models import Credit, ChatMessage

app = create_app()

with app.app_context():
    prof_id = 3
    
    total_purchased = db.session.query(db.func.sum(Credit.transaction_amount)).filter(
        Credit.professional_id == prof_id,
        Credit.transaction_type == 'purchase',
        Credit.payment_status == 'confirmed'
    ).scalar() or 0
    
    referral_bonuses = db.session.query(db.func.sum(Credit.transaction_amount)).filter(
        Credit.professional_id == prof_id,
        Credit.transaction_type == 'referral_bonus',
        Credit.payment_status == 'confirmed'
    ).scalar() or 0
    
    admin_additions = db.session.query(db.func.sum(Credit.transaction_amount)).filter(
        Credit.professional_id == prof_id,
        Credit.transaction_type == 'admin_addition',
        Credit.payment_status == 'confirmed'
    ).scalar() or 0
    
    total_credits = total_purchased + referral_bonuses + admin_additions
    
    used_credits = db.session.query(db.func.sum(ChatMessage.credits_used)).filter(
        ChatMessage.professional_id == prof_id
    ).scalar() or 0
    
    available = total_credits - used_credits
    
    print(f"Professional {prof_id} Credits:")
    print(f"  Purchased: {total_purchased}")
    print(f"  Referral: {referral_bonuses}")
    print(f"  Admin: {admin_additions}")
    print(f"  Total Added: {total_credits}")
    print(f"  Used: {used_credits}")
    print(f"  Available: {available}")
