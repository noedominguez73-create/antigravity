from app import create_app, db
from app.models import KnowledgeBaseDocument

app = create_app()

def list_docs():
    with app.app_context():
        print("=== KNOWLEDGE BASE DOCUMENTS ===")
        docs = KnowledgeBaseDocument.query.all()
        for doc in docs:
            print(f"ID: {doc.id} | ProfID: {doc.professional_id} | File: {doc.original_filename} | Type: {doc.file_type} | Size: {doc.file_size}")

if __name__ == "__main__":
    list_docs()
