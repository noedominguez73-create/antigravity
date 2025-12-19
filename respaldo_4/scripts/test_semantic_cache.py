import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import Professional, User, ChatbotConfig
from app.services.cache_service import cache_service

def test_caching():
    app = create_app()
    
    with app.app_context():
        print("[INFO] Iniciando Test de Semantic Caching...")
        
        # 1. Setup Data
        # Find or create a test professional
        prof = Professional.query.first()
        if not prof:
            print("[ERROR] No se encontro ningun profesional para probar.")
            return

        print(f"[INFO] Usando profesional: {prof.user.full_name} (ID: {prof.id})")
        
        # Ensure config exists
        config = ChatbotConfig.query.filter_by(professional_id=prof.id).first()
        if not config:
            config = ChatbotConfig(professional_id=prof.id, is_active=True)
            db.session.add(config)
            db.session.commit()
            
        # 2. Clear Cache for this test (Optional, but good for clean test)
        # We won't clear DB, but we can check if our query is already there.
        
        query = "Test de cache unico " + str(time.time())
        response_text = "Esta es una respuesta simulada para el cache."
        
        print(f"\n[1] Simulando Primera Consulta (Cache MISS)...")
        print(f"   Query: '{query}'")
        
        # Simulate adding to cache (as if Gemini returned it)
        start = time.time()
        cache_service.add_to_cache(query, response_text, prof.id)
        print(f"   [OK] Guardado en cache ({time.time() - start:.4f}s)")
        
        # 3. Test Exact Match
        print(f"\n[2] Probando Match Exacto...")
        start = time.time()
        cached = cache_service.get_cached_response(query, prof.id)
        elapsed = time.time() - start
        
        if cached == response_text:
            print(f"   [HIT] CACHE HIT! Respuesta correcta recuperada ({elapsed:.4f}s)")
        else:
            print(f"   [MISS] CACHE MISS o respuesta incorrecta: {cached}")

        # 4. Test Semantic Match (Similar Query)
        # Append some noise or change wording slightly
        similar_query = query + " por favor"
        print(f"\n[3] Probando Match Semantico (Query similar)...")
        print(f"   Query: '{similar_query}'")
        
        start = time.time()
        cached_sim = cache_service.get_cached_response(similar_query, prof.id)
        elapsed = time.time() - start
        
        if cached_sim == response_text:
            print(f"   [HIT] CACHE HIT SEMANTICO! Respuesta correcta recuperada ({elapsed:.4f}s)")
        else:
            print(f"   [MISS] CACHE MISS (Similitud insuficiente o error): {cached_sim}")
            
        # 5. Test Context Isolation (Different Professional)
        print(f"\n[4] Probando Aislamiento de Contexto (Otro ID)...")
        fake_prof_id = 99999
        cached_wrong = cache_service.get_cached_response(query, fake_prof_id)
        
        if cached_wrong is None:
            print("   [OK] CORRECTO: Cache ignorado para profesional incorrecto.")
        else:
            print("   [ERROR] ERROR: Se devolvio cache de otro profesional.")

if __name__ == "__main__":
    test_caching()
