import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import KnowledgeBaseDocument
from app.services.rag_service import rag_service

def ingest_all_documents():
    app = create_app()
    with app.app_context():
        print("[INFO] Starting RAG Ingestion...")
        
        documents = KnowledgeBaseDocument.query.all()
        
        if not documents:
            print("No documents found to ingest.")
            return

        print(f"Found {len(documents)} documents.")
        
        success_count = 0
        
        for doc in documents:
            print(f"Processing: {doc.original_filename} (ID: {doc.id})...")
            
            if not doc.text_content:
                print(f"   [SKIP] {doc.original_filename}: No text content.")
                continue
                
            success = rag_service.ingest_document(doc.id, doc.text_content)
            
            if success:
                print(f"   [OK] Ingested successfully.")
                success_count += 1
            else:
                print(f"   [FAIL] Failed to ingest.")
        
        print(f"\n[DONE] Ingestion Complete. {success_count}/{len(documents)} processed.")

if __name__ == "__main__":
    ingest_all_documents()
