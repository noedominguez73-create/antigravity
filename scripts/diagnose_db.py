import sqlite3
import os

# Path assumption based on app/__init__.py
db_path = 'c:/asesoriaimss.io/instance/asesoriaimss.db'

print(f"Checking database at: {db_path}")

if not os.path.exists(db_path):
    print("ERROR: Database file not found at expected path!")
    # Check parent dir
    print(f"Contents of instance dir: {os.listdir('c:/asesoriaimss.io/instance')}")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables found:")
        found_store = False
        for t in tables:
            print(f" - {t[0]}")
            if t[0] == 'store_types':
                found_store = True
                
        if found_store:
            print("SUCCESS: 'store_types' table exists.")
            # Check content
            cursor.execute("SELECT * FROM store_types")
            rows = cursor.fetchall()
            print(f"Content of 'store_types': {rows}")
        else:
            print("FAILURE: 'store_types' table MISSING.")
            
        conn.close()
    except Exception as e:
        print(f"Error accessing DB: {e}")
