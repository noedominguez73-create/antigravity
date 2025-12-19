import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import KnowledgeBaseChunk

app = create_app()
with app.app_context():
    print("Creating knowledge_base_chunks table...")
    try:
        KnowledgeBaseChunk.__table__.create(db.engine)
        print("Table created successfully.")
    except Exception as e:
        print(f"Error creating table (might already exist): {e}")
