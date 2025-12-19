import sqlite3
import os

DB_PATH = 'instance/asesoriaimss.db'
if not os.path.exists(DB_PATH):
    # Fallback path if instance folder structure is different
    DB_PATH = 'asesoriaimss.db'

print(f"Applying indices to database at: {DB_PATH}")

indices = [
    # Professionals
    "CREATE INDEX IF NOT EXISTS idx_professionals_user_id ON professionals(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_professionals_city ON professionals(city)",
    "CREATE INDEX IF NOT EXISTS idx_professionals_specialty ON professionals(specialty)",
    
    # Services
    "CREATE INDEX IF NOT EXISTS idx_services_professional_id ON services(professional_id)",
    
    # Experiences
    "CREATE INDEX IF NOT EXISTS idx_experiences_professional_id ON experiences(professional_id)",
    
    # Certifications
    "CREATE INDEX IF NOT EXISTS idx_certifications_professional_id ON certifications(professional_id)",
    
    # Comments
    "CREATE INDEX IF NOT EXISTS idx_comments_professional_id ON comments(professional_id)",
    "CREATE INDEX IF NOT EXISTS idx_comments_user_id ON users(id)", # Correction: user_id is in comments table
    "CREATE INDEX IF NOT EXISTS idx_comments_user_id_fk ON comments(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_comments_status ON comments(status)",
    
    # Credits
    "CREATE INDEX IF NOT EXISTS idx_credits_professional_id ON credits(professional_id)",
    "CREATE INDEX IF NOT EXISTS idx_credits_payment_status ON credits(payment_status)",
    
    # Referrals
    "CREATE INDEX IF NOT EXISTS idx_referrals_referrer_id ON referrals(referrer_id)",
    "CREATE INDEX IF NOT EXISTS idx_referrals_referred_user_id ON referrals(referred_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_referrals_code ON referrals(referral_code)",
    
    # Chat Messages
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_professional_id ON chat_messages(professional_id)",
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)"
]

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    for sql in indices:
        try:
            cursor.execute(sql)
            count += 1
            print(f"✅ Applied: {sql.split('ON')[1].strip()}")
        except Exception as e:
            print(f"⚠️ Error applying {sql}: {e}")
            
    conn.commit()
    conn.close()
    print(f"\nSuccessfully applied {count} indices.")
    
except Exception as e:
    print(f"❌ Critical Error: {e}")
