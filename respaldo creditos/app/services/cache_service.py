import numpy as np
import pickle
from app import db
from app.models import CachedResponse
import logging

logger = logging.getLogger(__name__)

class SemanticCacheService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SemanticCacheService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.initialized = True
        self.model_name = 'all-MiniLM-L6-v2'
        self.threshold = 0.85
        self.dimension = 384
        self.index = None
        self.encoder = None
        self.id_map = {} # Maps FAISS index to DB ID
        
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            logger.info(f"Loading Semantic Cache Model ({self.model_name})...")
            self.encoder = SentenceTransformer(self.model_name)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Load existing cache from DB
            self._load_cache_from_db()
            
        except ImportError:
            logger.error("Semantic Cache disabled: sentence-transformers or faiss-cpu not installed.")
        except Exception as e:
            logger.error(f"Error initializing Semantic Cache: {e}")

    def _load_cache_from_db(self):
        """Load all cached responses from DB into FAISS index"""
        try:
            # We need app context to query DB if not already present
            # Assuming this is called within app context usually
            from flask import current_app
            
            # If we are outside request context, we might need to handle it, 
            # but usually service is used inside routes.
            # However, __init__ might run on import. 
            # Let's wrap in try-catch for context issues.
            
            # Note: This is a heavy operation for startup if DB is huge.
            # For now it's fine.
            
            # We can't easily query DB here if outside context.
            # We'll lazy load or require explicit initialization if needed.
            # But for simplicity, let's assume we can query if context exists, 
            # or we skip and load on first request? 
            # Better: Just try, if fails, log warning.
            pass 
            
        except Exception as e:
            logger.warning(f"Could not load cache from DB on init: {e}")

    def reload_index(self):
        """Rebuild index from DB (Call this inside app context)"""
        if not self.encoder or not self.index:
            return

        try:
            responses = CachedResponse.query.all()
            self.index.reset()
            self.id_map = {}
            
            if not responses:
                return

            vectors = []
            valid_responses = []
            
            for r in responses:
                if r.embedding:
                    # Load pickle
                    emb = pickle.loads(r.embedding)
                    vectors.append(emb)
                    valid_responses.append(r)
                else:
                    # Compute embedding if missing (self-healing)
                    if self.encoder:
                        emb = self.encoder.encode([r.query])[0]
                        r.embedding = pickle.dumps(emb)
                        vectors.append(emb)
                        valid_responses.append(r)
            
            if vectors:
                # Add to FAISS
                vectors_np = np.array(vectors).astype('float32')
                self.index.add(vectors_np)
                
                # Update map with detached data (dicts) to avoid Session errors
                for i, r in enumerate(valid_responses):
                    self.id_map[i] = {
                        'id': r.id,
                        'professional_id': r.professional_id,
                        'response': r.response
                    }
            
            # Commit any self-healing updates
            db.session.commit()
            logger.info(f"Semantic Cache Index reloaded with {self.index.ntotal} entries.")
            
        except Exception as e:
            logger.error(f"Error reloading cache index: {e}")

    def get_cached_response(self, query, professional_id):
        """Search for cached response"""
        if not self.encoder or not self.index or self.index.ntotal == 0:
            return None
            
        try:
            # 1. Encode query
            query_vector = self.encoder.encode([query]).astype('float32')
            
            # 2. Search
            D, I = self.index.search(query_vector, 1)
            
            similarity = D[0][0]
            idx = I[0][0]
            
            if similarity > self.threshold:
                cached_item = self.id_map.get(idx)
                if cached_item:
                    # Verify professional_id matches (Context isolation)
                    if cached_item['professional_id'] == professional_id:
                        logger.info(f"Cache HIT ({similarity:.2f}) for query: {query}")
                        return cached_item['response']
                    else:
                        logger.info(f"Cache HIT but Professional Mismatch ({cached_item['professional_id']} != {professional_id})")
            
            return None
            
        except Exception as e:
            logger.error(f"Error in get_cached_response: {e}")
            return None

    def add_to_cache(self, query, response, professional_id):
        """Add new response to cache"""
        if not self.encoder or not self.index:
            return

        try:
            # 1. Encode
            vector = self.encoder.encode([query])[0]
            
            # 2. Save to DB
            new_cache = CachedResponse(
                professional_id=professional_id,
                query=query,
                response=response,
                embedding=pickle.dumps(vector)
            )
            db.session.add(new_cache)
            db.session.commit()
            
            # 3. Update In-Memory Index
            # FAISS add
            vector_np = np.array([vector]).astype('float32')
            self.index.add(vector_np)
            
            # Update map
            current_idx = self.index.ntotal - 1
            self.id_map[current_idx] = {
                'id': new_cache.id,
                'professional_id': new_cache.professional_id,
                'response': new_cache.response
            }
            
            logger.info(f"Added to cache: {query}")
            
        except Exception as e:
            logger.error(f"Error adding to cache: {e}")

# Singleton
cache_service = SemanticCacheService()
