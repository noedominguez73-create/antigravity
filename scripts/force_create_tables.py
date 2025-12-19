import sqlite3
from datetime import datetime

db_path = 'c:/asesoriaimss.io/instance/asesoriaimss.db'

print(f"Force creating tables in: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create store_types table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS store_types (
        id INTEGER PRIMARY KEY,
        name VARCHAR(100) NOT NULL UNIQUE,
        created_at DATETIME
    );
    """)
    print("Executed CREATE TABLE store_types")
    
    # 2. Create stores table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stores (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        store_type VARCHAR(100),
        store_name VARCHAR(200),
        created_at DATETIME,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    print("Executed CREATE TABLE stores")
    
    # 3. Create indices (optional but good)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stores_user_id ON stores (user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stores_store_type ON stores (store_type);")
    
    conn.commit()
    conn.close()
    print("SUCCESS: Tables created (or already existed).")
    
except Exception as e:
    print(f"ERROR: {e}")
