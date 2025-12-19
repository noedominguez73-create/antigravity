import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.services.rag_service import rag_service

def test_rag_retrieval():
    app = create_app()
    with app.app_context():
        print("[INFO] Testing RAG Retrieval...")
        
        # Reload index to ensure data is loaded
        rag_service.reload_index()
        
        # Use a query that should match the ingested document
        query = "tramite imss" 
        professional_id = 3 # Correct ID from DB check
        
        print(f"Query: '{query}'")
        
        context = rag_service.retrieve_context(query, professional_id)
        
        if context:
            print("\n[SUCCESS] Retrieved Context:")
            print("-" * 40)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 40)
        else:
            print("\n[WARNING] No context retrieved. (Is the index populated?)")

if __name__ == "__main__":
    test_rag_retrieval()
