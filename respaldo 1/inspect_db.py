from app import create_app, db
from app.models import KnowledgeBaseChunk, KnowledgeBaseDocument

app = create_app()

def inspect():
    with app.app_context():
        doc = KnowledgeBaseDocument.query.first()
        if not doc:
            print("No documents found.")
            return
            
        print(f"Document: {doc.original_filename}")
        
        chunk = KnowledgeBaseChunk.query.filter_by(document_id=doc.id, chunk_index=0).first()
        if chunk:
            print("=== CHUNK 0 CONTENT (Raw) ===")
            print(repr(chunk.content[:2000])) 
            print("=======================")
        else:
            print("No chunks found for this document.")

if __name__ == "__main__":
    inspect()
