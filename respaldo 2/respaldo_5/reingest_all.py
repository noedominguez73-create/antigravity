import os
from app import create_app, db
from app.models import KnowledgeBaseDocument, KnowledgeBaseChunk
from app.services.rag_service import rag_service

app = create_app()

def reingest():
    with app.app_context():
        print("=== RE-INGESTING DOCUMENTS ===")
        
        # 1. Clear Chunks
        print("Clearing existing chunks...")
        num_chunks = KnowledgeBaseChunk.query.delete()
        db.session.commit()
        print(f"Deleted {num_chunks} chunks.")
        
        # 2. Re-ingest
        docs = KnowledgeBaseDocument.query.all()
        print(f"Found {len(docs)} documents to re-ingest.")
        
        for doc in docs:
            print(f"Ingesting: {doc.original_filename} (ID: {doc.id})...")
            
            # Re-extract text to apply cleaning
            # chatbot.py uses 'uploads/knowledge_base' relative to CWD
            upload_folder = 'uploads/knowledge_base'
            file_path = os.path.join(upload_folder, doc.filename)
            
            if os.path.exists(file_path):
                print(f"  Re-extracting text from {file_path}...")
                from app.utils.file_extractor import FileExtractor
                new_text = FileExtractor.extract_text(file_path, doc.file_type)
                
                # Update DB
                doc.text_content = new_text
                db.session.add(doc)
                
                # Ingest
                rag_service.ingest_document(doc.id, new_text)
            else:
                print(f"  [WARNING] File not found: {file_path}. Using existing text.")
                rag_service.ingest_document(doc.id, doc.text_content)
            
        db.session.commit()
        print("Done!")

if __name__ == "__main__":
    reingest()
