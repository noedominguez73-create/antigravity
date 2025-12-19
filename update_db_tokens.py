import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), 'instance', 'asesoriaimss.db')

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if columns exist
        cursor.execute("PRAGMA table_info(mirror_usages)")
        columns = [info[1] for info in cursor.fetchall()]

        new_columns = {
            'prompt_tokens': 'INTEGER DEFAULT 0',
            'completion_tokens': 'INTEGER DEFAULT 0',
            'total_tokens': 'INTEGER DEFAULT 0'
        }

        for col, dtype in new_columns.items():
            if col not in columns:
                print(f"Adding column {col}...")
                cursor.execute(f"ALTER TABLE mirror_usages ADD COLUMN {col} {dtype}")
            else:
                print(f"Column {col} already exists.")

        conn.commit()
        print("Migration completed successfully.")

    except Exception as e:
        print(f"Error migrating database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
