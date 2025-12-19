
import sys
import os
from app import create_app, db
from app.models import Professional, KnowledgeBaseDocument, KnowledgeBaseChunk
from app.services.rag_service import rag_service

def debug_rag(professional_id):
    app = create_app()
    with app.app_context():
        print(f"\n=== DEBUGGING RAG FOR PROFESSIONAL ID: {professional_id} ===")
        
        # 1. Check Documents
        docs = KnowledgeBaseDocument.query.filter_by(professional_id=professional_id).all()
        print(f"\n[DOCUMENTS FOUND: {len(docs)}]")
        
        for doc in docs:
            print(f"\n>>> Document ID: {doc.id} | Name: {doc.original_filename} | Size: {doc.file_size} <<<")
            
            # Check chunks for this doc
            chunks_count = KnowledgeBaseChunk.query.filter_by(document_id=doc.id).count()
            print(f"    Chunks in DB: {chunks_count}")
            
            # Preview first chunk
            first_chunk = KnowledgeBaseChunk.query.filter_by(document_id=doc.id).order_by(KnowledgeBaseChunk.chunk_index).first()
            if first_chunk:
                print(f"    First Chunk Preview: {repr(first_chunk.content[:100])}")
            
            # Search for Article 3
            print(f"    Searching for 'Artículo 3' in doc {doc.id}...")
            art3_chunk = KnowledgeBaseChunk.query.filter(
                KnowledgeBaseChunk.document_id == doc.id,
                KnowledgeBaseChunk.content.ilike("%Artículo 3%")
            ).first()
            
            if art3_chunk:
                print(f"    FOUND 'Artículo 3': {repr(art3_chunk.content[:200])}")
            else:
                print("    'Artículo 3' NOT FOUND in this document.")

        # 4. Test Retrieval with specific query
        query = "Qué dice el artículo tercero"
        print(f"\n[TESTING RETRIEVAL FOR QUERY: '{query}']")
        try:
            rag_service.reload_index()
            # Use the internal search method to get details
            query_vector = rag_service.encoder.encode([query]).astype('float32')
            fetch_k = 100 * 100 # Fetch deep
            D, I = rag_service.index.search(query_vector, fetch_k)
            
            print(f"\n[SEARCHING FOR 'Artículo 3' IN TOP 100]")
            found_at = -1
            count = 0
            
            for rank, idx in enumerate(I[0]):
                if idx == -1: continue
                chunk_id = rag_service.id_map.get(idx)
                if chunk_id:
                    chunk = KnowledgeBaseChunk.query.get(chunk_id)
                    if chunk:
                        doc = KnowledgeBaseDocument.query.get(chunk.document_id)
                        if doc and doc.professional_id == professional_id:
                            count += 1
                            if "Artículo 3" in chunk.content:
                                print(f"  >>> FOUND 'Artículo 3' at Rank {count} (Score={D[0][rank]:.4f}) <<<")
                                print(f"  Content: {chunk.content[:200]}...")
                                found_at = count
                                break
                            
                            if count >= 100:
                                break
            
            if found_at == -1:
                print("  'Artículo 3' NOT FOUND in top 100 results.")
        except Exception as e:
            print(f"  ERROR during retrieval: {e}")

if __name__ == "__main__":
    # Default to ID 3 based on screenshots, or take arg
    prof_id = 3
    if len(sys.argv) > 1:
        prof_id = int(sys.argv[1])
        
    debug_rag(prof_id)
