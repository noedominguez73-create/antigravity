import numpy as np
import pickle
from app import db
from app.models import KnowledgeBaseChunk, KnowledgeBaseDocument
import logging

logger = logging.getLogger(__name__)

class RAGService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.initialized = True
        self.model_name = 'all-MiniLM-L6-v2'
        self.dimension = 384
        self.encoder = None
        self.index = None
        self.id_map = {} # Maps FAISS index to DB ID (Chunk ID)
        
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            logger.info(f"Loading RAG Model ({self.model_name})...")
            self.encoder = SentenceTransformer(self.model_name)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Load existing chunks from DB
            self._load_chunks_from_db()
            
        except ImportError:
            logger.error("RAG Service disabled: sentence-transformers or faiss-cpu not installed.")
        except Exception as e:
            logger.error(f"Error initializing RAG Service: {e}")

    def _load_chunks_from_db(self):
        """Load all chunks from DB into FAISS index"""
        try:
            # Lazy load inside app context usually
            pass
        except Exception as e:
            logger.warning(f"Could not load RAG chunks from DB on init: {e}")

    def reload_index(self):
        """Rebuild index from DB (Call this inside app context)"""
        if not self.encoder or not self.index:
            return

        try:
            chunks = KnowledgeBaseChunk.query.all()
            self.index.reset()
            self.id_map = {}
            
            if not chunks:
                return

            vectors = []
            valid_chunks = []
            
            for chunk in chunks:
                if chunk.embedding:
                    emb = pickle.loads(chunk.embedding)
                    vectors.append(emb)
                    valid_chunks.append(chunk)
            
            if vectors:
                vectors_np = np.array(vectors).astype('float32')
                self.index.add(vectors_np)
                
                for i, chunk in enumerate(valid_chunks):
                    self.id_map[i] = chunk.id
            
            logger.info(f"RAG Index reloaded with {self.index.ntotal} chunks.")
            
        except Exception as e:
            logger.error(f"Error reloading RAG index: {e}")

    def ingest_document(self, document_id, text_content, chunk_size=500):
        """Split document into chunks and save to DB"""
        if not self.encoder or not text_content:
            return False

        try:
            # 1. Split text into chunks (Simple character split for now, better to use RecursiveCharacterTextSplitter)
            # Approximate 4 chars per token -> 500 tokens ~= 2000 chars
            # Approximate 4 chars per token -> 500 tokens ~= 2000 chars
            chars_per_chunk = chunk_size * 4
            overlap = 200 # 50 tokens overlap
            
            chunks = []
            start = 0
            while start < len(text_content):
                end = start + chars_per_chunk
                chunks.append(text_content[start:end])
                start += (chars_per_chunk - overlap)
            
            # 2. Process chunks
            # Batch encode for performance
            embeddings = self.encoder.encode(chunks)
            
            for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
                # Save to DB
                kb_chunk = KnowledgeBaseChunk(
                    document_id=document_id,
                    content=chunk_text,
                    embedding=pickle.dumps(vector),
                    chunk_index=i
                )
                db.session.add(kb_chunk)
            
            db.session.commit()
            
            # 3. Update Index
            self.reload_index()
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting document {document_id}: {e}")
            db.session.rollback()
            return False

    def retrieve_context(self, query, professional_id, top_k=5):
        """Retrieve most relevant chunks for a query"""
        if not self.encoder or not self.index or self.index.ntotal == 0:
            return ""
            
        try:
            # Debug log entry
            with open('rag_debug.log', 'a', encoding='utf-8') as f:
                f.write(f"\n--- RETRIEVE CONTEXT CALLED ---\nQuery: '{query}'\nProfID: {professional_id}\n")

            # 1. Encode query
            query_vector = self.encoder.encode([query]).astype('float32')
            
            # 2. Search (Vector)
            # Fetch more candidates to allow for filtering by professional_id
            fetch_k = top_k * 100 
            D, I = self.index.search(query_vector, fetch_k)
            
            print(f"RAG Search: Query='{query}', ProfID={professional_id}, FetchK={fetch_k}")
            
            relevant_chunks = []
            seen_chunk_ids = set()

            # --- KEYWORD BOOST (Hybrid Search) ---
            # If query asks for specific article, force include it
            import re
            
            # Map text numbers to digits
            text_to_digit = {
                'primero': '1', 'uno': '1', '1ro': '1',
                'segundo': '2', 'dos': '2', '2do': '2',
                'tercero': '3', 'tres': '3', '3ro': '3',
                'cuarto': '4', 'cuatro': '4', '4to': '4',
                'quinto': '5', 'cinco': '5', '5to': '5',
                'sexto': '6', 'seis': '6', '6to': '6',
                'septimo': '7', 'siete': '7', '7mo': '7', 'séptimo': '7',
                'octavo': '8', 'ocho': '8', '8vo': '8',
                'noveno': '9', 'nueve': '9', '9no': '9',
                'decimo': '10', 'diez': '10', '10mo': '10', 'décimo': '10'
            }
            
            # Match "articulo X" where X can be digit or text
            match = re.search(r'art[íi]culo\s+(\w+)', query.lower())
            if match:
                val = match.group(1)
                art_num = None
                
                if val.isdigit():
                    art_num = val
                elif val in text_to_digit:
                    art_num = text_to_digit[val]
                
                if art_num:
                    print(f"  Detected search for Article {art_num} (from '{val}'). Boosting...")
                    
                    # Search DB for "Artículo X" or "Articulo X"
                    keyword_chunks = KnowledgeBaseChunk.query.filter(
                        (KnowledgeBaseChunk.content.ilike(f"%Artículo {art_num}%")) | 
                        (KnowledgeBaseChunk.content.ilike(f"%Articulo {art_num}%"))
                    ).all()
                    
                    for k_chunk in keyword_chunks:
                        # Filter by professional
                        doc = KnowledgeBaseDocument.query.get(k_chunk.document_id)
                        if doc and doc.professional_id == professional_id:
                            # Prioritize chunks that START with the article to find definition
                            # e.g. "Artículo 3. ..." or "Artículo 3.-"
                            # Check first 50 chars for loose match and use LOWERCASE
                            content_start = k_chunk.content[:50].lower()
                            target = f"artículo {art_num}"
                            target_noacc = f"articulo {art_num}"
                            
                            # Debug log
                            with open('rag_debug.log', 'a', encoding='utf-8') as f:
                                f.write(f"Checking Chunk {k_chunk.id}: '{content_start}' vs '{target}'/'{target_noacc}'\n")
                            
                            if target in content_start or target_noacc in content_start:
                                 with open('rag_debug.log', 'a', encoding='utf-8') as f:
                                     f.write(f"  >>> BOOSTING CHUNK {k_chunk.id} <<<\n")
                                 print(f"  >>> BOOSTING DEFINITION CHUNK {k_chunk.id} <<<")
                                 relevant_chunks.insert(0, k_chunk.content) # Put at TOP
                            else:
                                 relevant_chunks.append(k_chunk.content)
                            
                            seen_chunk_ids.add(k_chunk.id)
                            
                            if len(relevant_chunks) >= top_k: 
                                break

            # --- VECTOR SEARCH ---
            for rank, idx in enumerate(I[0]):
                if idx == -1: continue
                
                chunk_id = self.id_map.get(idx)
                if chunk_id and chunk_id not in seen_chunk_ids:
                    chunk = KnowledgeBaseChunk.query.get(chunk_id)
                    if chunk:
                        # Filter by professional_id (via document)
                        doc = KnowledgeBaseDocument.query.get(chunk.document_id)
                        if doc and doc.professional_id == professional_id:
                            # print(f"  Match Found: Rank={rank}, ChunkID={chunk.id}, Score={D[0][rank]:.4f}")
                            relevant_chunks.append(chunk.content)
                            seen_chunk_ids.add(chunk.id)
                            
                            if len(relevant_chunks) >= top_k:
                                break
            
            print(f"RAG Search Result: Found {len(relevant_chunks)} chunks")
            return "\n\n---\n\n".join(relevant_chunks)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

# Singleton
rag_service = RAGService()
