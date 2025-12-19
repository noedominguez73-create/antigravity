from app import create_app, db
from app.models import KnowledgeBaseDocument, KnowledgeBaseChunk
from app.services.rag_service import rag_service

app = create_app()

with app.app_context():
    print("Checking for unindexed documents...")
    
    docs = KnowledgeBaseDocument.query.all()
    indexed_count = 0
    
    for doc in docs:
        # Check if chunks exist
        chunk_count = KnowledgeBaseChunk.query.filter_by(document_id=doc.id).count()
        
        if chunk_count == 0:
            print(f"Indexing document: {doc.filename} (ID: {doc.id})...")
            if doc.text_content:
                success = rag_service.ingest_document(doc.id, doc.text_content)
                if success:
                    print(f"  -> Successfully indexed.")
                    indexed_count += 1
                else:
                    print(f"  -> Failed to index.")
            else:
                print(f"  -> Skipped (No text content).")
        else:
            print(f"Document {doc.filename} (ID: {doc.id}) already has {chunk_count} chunks.")
            
    print(f"\nFinished. Indexed {indexed_count} new documents.")
