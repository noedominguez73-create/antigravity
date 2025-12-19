import sqlite3
import os

def inspect_db():
    db_path = 'instance/asesoriaimss.db'
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("="*50)
    print("DATABASE INSPECTION REPORT")
    print("="*50)

    # 1. Check Foreign Keys Support
    cursor.execute("PRAGMA foreign_keys;")
    fk_status = cursor.fetchone()[0]
    print(f"Foreign Keys Enabled: {'YES' if fk_status else 'NO'}")
    
    # 2. List Tables
    print("\n[TABLES]")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        print(f"- {table[0]}")

    # 3. Detailed Table Inspection
    print("\n[DETAILED SCHEMA & INDICES]")
    for table in tables:
        table_name = table[0]
        if table_name == 'sqlite_sequence': continue
        
        print(f"\n--- Table: {table_name} ---")
        
        # Columns & Constraints
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            # cid, name, type, notnull, dflt_value, pk
            pk = "PK" if col[5] else ""
            nn = "NOT NULL" if col[3] else "NULL"
            print(f"  {col[1]:<20} {col[2]:<10} {nn:<10} {pk}")

        # Foreign Keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        fks = cursor.fetchall()
        if fks:
            print("Foreign Keys:")
            for fk in fks:
                # id, seq, table, from, to, on_update, on_delete, match
                print(f"  {fk[3]} -> {fk[2]}({fk[4]}) ON DELETE {fk[6]} ON UPDATE {fk[5]}")
        
        # Indices
        cursor.execute(f"PRAGMA index_list({table_name})")
        indices = cursor.fetchall()
        if indices:
            print("Indices:")
            for idx in indices:
                # seq, name, unique, origin, partial
                unique = "UNIQUE" if idx[2] else ""
                print(f"  {idx[1]} {unique}")
                
                # Index Info
                cursor.execute(f"PRAGMA index_info({idx[1]})")
                idx_cols = cursor.fetchall()
                cols = ", ".join([c[2] for c in idx_cols])
                print(f"    -> Columns: {cols}")
        else:
            print("Indices: NONE (Only PK)")

    conn.close()

if __name__ == "__main__":
    inspect_db()
