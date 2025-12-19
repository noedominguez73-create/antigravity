import sys
import os
import time

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.services.cache_service import cache_service
from app.models import CachedResponse

def test_semantic_cache():
    app = create_app()
    with app.app_context():
        print("[INFO] Testing Semantic Cache...")
        
        try:
            # 1. Clear existing cache for test professional
            prof_id = 999
            # Use db.session.query for robustness
            db.session.query(CachedResponse).filter_by(professional_id=prof_id).delete()
            db.session.commit()
            
            # Reload index to ensure it's clean
            cache_service.reload_index()
            
            # 2. Add to cache
            query = "requisitos modalidad 40"
            response = "Para modalidad 40 necesitas 52 semanas cotizadas y no tener relacion laboral."
            print(f"[INFO] Adding to cache: '{query}'")
            cache_service.add_to_cache(query, response, prof_id)
            
            # 3. Test Exact Match
            print("\n[INFO] Testing Exact Match...")
            res = cache_service.get_cached_response(query, prof_id)
            if res == response:
                print("[SUCCESS] Exact match found!")
            else:
                print(f"[FAIL] Exact match FAILED. Got: {res}")
                
            # 4. Test Semantic Match
            similar_query = "cuales son los requisitos para la mod 40"
            print(f"\n[INFO] Testing Semantic Match: '{similar_query}'")
            res = cache_service.get_cached_response(similar_query, prof_id)
            if res == response:
                print("[SUCCESS] Semantic match found!")
            else:
                print(f"[FAIL] Semantic match FAILED. Got: {res}")
                
            # 5. Test Miss
            diff_query = "como me doy de alta en la clinica"
            print(f"\n[INFO] Testing Miss: '{diff_query}'")
            res = cache_service.get_cached_response(diff_query, prof_id)
            if res is None:
                print("[SUCCESS] Correctly missed!")
            else:
                print(f"[FAIL] Should have missed but got: {res}")
                
            # 6. Test Persistence (Reload Index)
            print("\n[INFO] Testing Persistence (Reloading Index)...")
            cache_service.reload_index()
            res = cache_service.get_cached_response(similar_query, prof_id)
            if res == response:
                print("[SUCCESS] Persistence verified!")
            else:
                print(f"[FAIL] Persistence FAILED. Got: {res}")

            # Cleanup
            db.session.query(CachedResponse).filter_by(professional_id=prof_id).delete()
            db.session.commit()
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_semantic_cache()
