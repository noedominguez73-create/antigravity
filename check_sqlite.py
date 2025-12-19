import sqlite3
import os

db_path = 'instance/asesoriaimss.db'

print("=" * 60)
print("SQLite Database Check")
print("=" * 60)
print(f"Database path: {db_path}")
print(f"Absolute path: {os.path.abspath(db_path)}")
print(f"File exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    print(f"File size: {os.path.getsize(db_path)} bytes")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\nTables in database: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")
            
        # Count users
        cursor.execute("SELECT COUNT(*) FROM users;")
        user_count = cursor.fetchone()[0]
        print(f"\nUsers in database: {user_count}")
        
        # List users
        cursor.execute("SELECT id, email, full_name, role FROM users;")
        users = cursor.fetchall()
        print("\nUser details:")
        for user in users:
            print(f"  ID: {user[0]}, Email: {user[1]}, Name: {user[2]}, Role: {user[3]}")
        
        conn.close()
        print("\n[OK] Database is accessible and contains data!")
        
    except Exception as e:
        print(f"\n[ERROR] Database error: {str(e)}")
else:
    print("\n[ERROR] Database file does not exist!")

print("=" * 60)
