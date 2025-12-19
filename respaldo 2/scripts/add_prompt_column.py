from app import create_app, db
from sqlalchemy import text

app = create_app()

def add_prompt_column():
    with app.app_context():
        # Check if column exists
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(mirror_items)"))
            columns = [row[1] for row in result.fetchall()]
            
            if 'prompt' not in columns:
                print("Adding 'prompt' column to mirror_items table...")
                try:
                    conn.execute(text("ALTER TABLE mirror_items ADD COLUMN prompt TEXT"))
                    conn.commit()
                    print("Column added successfully.")
                except Exception as e:
                    print(f"Error adding column: {e}")
            else:
                print("'prompt' column already exists.")

if __name__ == '__main__':
    add_prompt_column()
