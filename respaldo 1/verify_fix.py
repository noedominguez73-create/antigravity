
import sys
from app import create_app
from app.services.rag_service import rag_service

def verify_fix(professional_id=3):
    app = create_app()
    with app.app_context():
        with open('verify_log.txt', 'w', encoding='utf-8') as f:
            def log(msg):
                f.write(msg + "\n")
                print(msg)
            
            log(f"\n=== VERIFYING RAG FIX FOR PROFESSIONAL ID: {professional_id} ===")
            
            # Force reload index
            rag_service.reload_index()
            
            queries = [
                "Qué dice el artículo tercero",
                "dime el articulo 3",
                "articulo 3"
            ]
            
            for query in queries:
                log(f"\nTesting Query: '{query}'")
                results = rag_service.retrieve_context(query, professional_id, top_k=3)
                
                log("-" * 40)
                log(results[:500] + "...") 
                log("-" * 40)
                
                # Check for Article 3 at the start (ignoring case/accents for check)
                res_start = results[:100].lower()
                if "artículo 3" in res_start or "articulo 3" in res_start:
                    log(">>> SUCCESS: Article 3 found at the TOP! <<<")
                else:
                    log(">>> FAILURE: Article 3 NOT found at the top. <<<")

if __name__ == "__main__":
    verify_fix()
