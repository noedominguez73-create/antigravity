import numpy as np
import time
import sys

# Nota: Requiere instalar librerías:
# pip install sentence-transformers faiss-cpu

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("❌ Error: Librerías faltantes.")
    print("Por favor instala: pip install sentence-transformers faiss-cpu")
    sys.exit(1)

class SemanticCache:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.85):
        print(f"🔄 Cargando modelo de embeddings ({model_name})...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = 384 # Dimensión para all-MiniLM-L6-v2
        
        # Índice FAISS para búsqueda de similitud (Inner Product)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Almacenamiento de respuestas (en memoria para demo, usar Redis/DB en prod)
        self.responses = {}
        self.queries = []
        self.threshold = threshold

    def get_response(self, query):
        """Busca una respuesta en cache similar a la query"""
        if self.index.ntotal == 0:
            return None
            
        # 1. Generar embedding
        start_time = time.time()
        query_vector = self.encoder.encode([query])
        
        # 2. Buscar vecino más cercano
        # D = Distancias (Similitudes), I = Índices
        D, I = self.index.search(query_vector, 1)
        
        similarity = D[0][0]
        idx = I[0][0]
        elapsed = time.time() - start_time
        
        print(f"🔍 Búsqueda: '{query}' -> Mejor match índice {idx} con similitud {similarity:.4f} ({elapsed:.3f}s)")
        
        if similarity > self.threshold:
            print(f"✅ CACHE HIT! (Similitud > {self.threshold})")
            return self.responses[idx]
        
        print(f"❌ CACHE MISS (Similitud < {self.threshold})")
        return None

    def add_response(self, query, response):
        """Guarda una nueva respuesta en el cache"""
        print(f"💾 Guardando en cache: '{query}'")
        vector = self.encoder.encode([query])
        self.index.add(vector)
        
        # El índice del nuevo elemento es ntotal - 1
        current_idx = self.index.ntotal - 1
        self.responses[current_idx] = response
        self.queries.append(query)

# ==========================================
# DEMOSTRACIÓN
# ==========================================
if __name__ == "__main__":
    print("🚀 Iniciando Demo de Semantic Caching")
    print("=====================================")
    
    cache = SemanticCache(threshold=0.80)
    
    # 1. Primera consulta (Simulando llamada a Gemini)
    q1 = "¿Cuáles son los requisitos para la Modalidad 40?"
    r1 = """
    Para la Modalidad 40 del IMSS necesitas:
    1. No tener relación laboral vigente.
    2. Tener al menos 52 semanas cotizadas en los últimos 5 años.
    3. Solicitud por escrito.
    """
    
    print(f"\nUsuario pregunta: {q1}")
    result = cache.get_response(q1)
    
    if not result:
        print("🤖 Llamando a Gemini API... (Simulado)")
        cache.add_response(q1, r1)
    
    # 2. Consulta idéntica
    print(f"\nUsuario pregunta (Idéntica): {q1}")
    result = cache.get_response(q1)
    if result:
        print(f"🤖 Respuesta desde Cache: {result[:50]}...")

    # 3. Consulta semánticamente similar (fraseo diferente)
    q2 = "dime los requisitos para entrar a modalidad 40"
    print(f"\nUsuario pregunta (Similar): {q2}")
    result = cache.get_response(q2)
    if result:
        print(f"🤖 Respuesta desde Cache: {result[:50]}...")
        
    # 4. Consulta diferente
    q3 = "¿Cómo me doy de alta en la clínica?"
    print(f"\nUsuario pregunta (Diferente): {q3}")
    result = cache.get_response(q3)
    if not result:
        print("🤖 Llamando a Gemini API... (Simulado)")
        # No guardamos para no ensuciar el demo
