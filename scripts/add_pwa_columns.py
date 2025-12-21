import sqlite3
import os

# Path based on app/__init__.py: os.path.join(app.instance_path, 'asesoriaimss.db')
# Assuming script is run from project root, instance is often at root or app/instance
DB_PATHS = [
    'instance/asesoriaimss.db',
    'app/instance/asesoriaimss.db',
    'asesoriaimss.db'
]

def migrate():
    db_path = None
    for path in DB_PATHS:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("Error: Database not found in common locations.")
        return

    print(f"Migrating database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Add phone_number
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)")
            print("Added column: phone_number")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("Column phone_number already exists.")
            else:
                print(f"Error adding phone_number: {e}")

        # Add push_subscription
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN push_subscription TEXT")
            print("Added column: push_subscription")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("Column push_subscription already exists.")
            else:
                print(f"Error adding push_subscription: {e}")

        conn.commit()
        print("Migration complete.")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
