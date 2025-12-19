import sqlite3
import os

DB_PATH = os.path.join(os.getcwd(), 'instance', 'asesoriaimss.db')

def verify_schema():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA table_info(mirror_usages)")
        columns = [info[1] for info in cursor.fetchall()]
        
        expected_columns = ['prompt_tokens', 'completion_tokens', 'total_tokens']
        missing_columns = [col for col in expected_columns if col not in columns]

        if not missing_columns:
            print("SUCCESS: All token columns found in mirror_usages.")
            print(f"Columns present: {columns}")
        else:
            print(f"FAILURE: Missing columns: {missing_columns}")

    except Exception as e:
        print(f"Error checking schema: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    verify_schema()
