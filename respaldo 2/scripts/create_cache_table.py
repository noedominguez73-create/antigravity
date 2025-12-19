import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import CachedResponse

app = create_app()
with app.app_context():
    print("Creating cached_responses table...")
    try:
        CachedResponse.__table__.create(db.engine)
        print("Table created successfully.")
    except Exception as e:
        print(f"Error creating table (might already exist): {e}")
