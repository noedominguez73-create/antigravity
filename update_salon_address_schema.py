
from app import create_app, db
from sqlalchemy import text

app = create_app()

def add_column_if_not_exists(table, column, type_def):
    with app.app_context():
        with db.engine.connect() as conn:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {type_def}"))
                print(f"Added column {column} to {table}")
            except Exception as e:
                if "duplicate column" in str(e).lower() or "no such table" in str(e).lower() or "already exists" in str(e).lower():
                    print(f"Column {column} likely already exists in {table}. ({str(e)})")
                else:
                    print(f"Error adding {column}: {e}")

if __name__ == "__main__":
    print("Updating SalonConfig schema for address segmentation...")
    add_column_if_not_exists('salon_configs', 'city', 'VARCHAR(100)')
    add_column_if_not_exists('salon_configs', 'state', 'VARCHAR(100)')
    add_column_if_not_exists('salon_configs', 'country', 'VARCHAR(100) DEFAULT "México"')
    print("Schema update complete.")
