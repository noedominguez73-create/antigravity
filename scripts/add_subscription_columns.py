import sqlite3
import os

# Path based on app/__init__.py
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
        print("Error: Database not found.")
        return

    print(f"Migrating database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    columns = [
        ("subscription_status", "VARCHAR(20) DEFAULT 'active'"),
        ("subscription_end_date", "DATETIME"),
        ("monthly_token_limit", "INTEGER DEFAULT 1000"),
        ("current_month_tokens", "INTEGER DEFAULT 0")
    ]

    for col_name, col_type in columns:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            print(f"Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print(f"Column {col_name} already exists.")
            else:
                print(f"Error adding {col_name}: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
