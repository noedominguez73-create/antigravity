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
            chars_per_chunk = chunk_size * 4
            chunks = [text_content[i:i+chars_per_chunk] for i in range(0, len(text_content), chars_per_chunk)]
            
            # 2. Process chunks
            for i, chunk_text in enumerate(chunks):
                # Generate embedding
                vector = self.encoder.encode([chunk_text])[0]
                
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

    def retrieve_context(self, query, professional_id, top_k=3):
        """Retrieve most relevant chunks for a query"""
        if not self.encoder or not self.index or self.index.ntotal == 0:
            return ""
            
        try:
            # 1. Encode query
            query_vector = self.encoder.encode([query]).astype('float32')
            
            # 2. Search
            D, I = self.index.search(query_vector, top_k * 2) # Fetch more to filter
            
            relevant_chunks = []
            
            for idx in I[0]:
                if idx == -1: continue
                
                chunk_id = self.id_map.get(idx)
                if chunk_id:
                    chunk = KnowledgeBaseChunk.query.get(chunk_id)
                    if chunk:
                        # Filter by professional_id (via document)
                        doc = KnowledgeBaseDocument.query.get(chunk.document_id)
                        if doc and doc.professional_id == professional_id:
                            relevant_chunks.append(chunk.content)
                            if len(relevant_chunks) >= top_k:
                                break
            
            return "\n\n---\n\n".join(relevant_chunks)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

# Singleton
rag_service = RAGService()
