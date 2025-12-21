import sys
import os

# Fix path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import SalonConfig
import sqlite3

app = create_app()

def migrate():
    with app.app_context():
        # Get DB path logic from config or assume standard
        db_path = os.path.join(app.instance_path, 'asesoriaimss.db')
        print(f"Migrating DB at: {db_path}")
        
        # Helper to check if table exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='salon_configs'")
        exists = cursor.fetchone()
        
        conn.close()
        
        if not exists:
            print("Creating 'salon_configs' table...")
            db.create_all() # This creates all missing tables including our new one
            print("Migration successful: Table 'salon_configs' created.")
        else:
            print("Table 'salon_configs' already exists. Examining columns...")
            # If we needed to add columns incrementally we would do it here
            pass
            
if __name__ == '__main__':
    migrate()
