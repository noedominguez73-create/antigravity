from app import create_app, db
from app.models import KnowledgeBaseChunk, KnowledgeBaseDocument

app = create_app()

def dump_chunks():
    with app.app_context():
        # Target the TXT file (ID 1 based on previous output)
        doc_id = 1
        doc = KnowledgeBaseDocument.query.get(doc_id)
        if not doc:
            print(f"Document {doc_id} not found.")
            return
            
        print(f"Dumping chunks for: {doc.original_filename}")
        
        chunks = KnowledgeBaseChunk.query.filter_by(document_id=doc.id).order_by(KnowledgeBaseChunk.chunk_index).all()
        
        with open('chunks_dump.txt', 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(f"\n=== CHUNK {chunk.chunk_index} ===\n")
                f.write(chunk.content)
                f.write("\n=======================\n")
                
        print(f"Dumped {len(chunks)} chunks to chunks_dump.txt")

if __name__ == "__main__":
    dump_chunks()
