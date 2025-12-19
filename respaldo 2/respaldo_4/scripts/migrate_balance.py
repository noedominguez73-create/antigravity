import sys
import os
from sqlalchemy import text

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import Professional, Credit, ChatMessage

def migrate_balance():
    app = create_app()
    with app.app_context():
        print("🚀 Starting Balance Migration...")
        
        # 1. Add Column (SQLite specific)
        # Check if column exists first to avoid error
        try:
            with db.engine.connect() as conn:
                result = conn.execute(text("PRAGMA table_info(professionals)"))
                columns = [row[1] for row in result]
                
                if 'balance' not in columns:
                    print("Adding 'balance' column to professionals table...")
                    conn.execute(text("ALTER TABLE professionals ADD COLUMN balance INTEGER DEFAULT 0"))
                    conn.commit()
                    print("Column added.")
                else:
                    print("Column 'balance' already exists.")
        except Exception as e:
            print(f"Error adding column: {e}")
            return

        # 2. Calculate and Populate Balance
        print("Calculating historical balances...")
        professionals = Professional.query.all()
        
        for prof in professionals:
            # Sum Credits Added
            total_purchased = db.session.query(db.func.sum(Credit.transaction_amount)).filter(
                Credit.professional_id == prof.id,
                Credit.payment_status == 'confirmed'
            ).scalar() or 0
            
            # Sum Credits Used
            total_used = db.session.query(db.func.sum(ChatMessage.credits_used)).filter(
                ChatMessage.professional_id == prof.id
            ).scalar() or 0
            
            # Calculate Balance
            current_balance = total_purchased - total_used
            
            # Update Professional
            prof.balance = current_balance
            print(f"   Prof {prof.id} ({prof.user.full_name}): +{total_purchased} -{total_used} = {current_balance}")
        
        try:
            db.session.commit()
            print("✅ Migration completed successfully. Balances updated.")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error updating balances: {e}")

if __name__ == "__main__":
    migrate_balance()
