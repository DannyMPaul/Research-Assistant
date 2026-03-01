"""Optimized vector store implementation with better memory management and performance."""
import numpy as np
import json
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import re
from collections import Counter
import math
import logging
from config import settings

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers or faiss not available. Vector search disabled.")

logger = logging.getLogger(__name__)

class ChunkManager:
    """Manages document chunking with optimized memory usage."""
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text with optimized memory usage."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks

class TextProcessor:
    """Handles text preprocessing with caching."""
    def __init__(self, cache_size: int = 1000):
        self.preprocess_cache: Dict[str, str] = {}
        self.tokenize_cache: Dict[str, List[str]] = {}
        self.cache_size = cache_size

    def preprocess_text(self, text: str) -> str:
        """Preprocess text with caching."""
        if text in self.preprocess_cache:
            return self.preprocess_cache[text]

        # Clean and normalize text
        processed = text.lower()
        processed = re.sub(r'\s+', ' ', processed)
        processed = re.sub(r'[^\w\s]', ' ', processed)
        processed = processed.strip()

        # Update cache with LRU policy
        if len(self.preprocess_cache) >= self.cache_size:
            self.preprocess_cache.pop(next(iter(self.preprocess_cache)))
        self.preprocess_cache[text] = processed

        return processed

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with caching."""
        if text in self.tokenize_cache:
            return self.tokenize_cache[text]

        tokens = [token for token in text.split() if len(token) > 1]

        if len(self.tokenize_cache) >= self.cache_size:
            self.tokenize_cache.pop(next(iter(self.tokenize_cache)))
        self.tokenize_cache[text] = tokens

        return tokens

class VectorStore:
    """Optimized vector store with improved memory management and search capabilities."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = settings.VECTOR_DIMENSION
        self.vector_store_path = settings.VECTOR_STORE_DIR
        
        # Initialize components
        self.chunk_manager = ChunkManager()
        self.text_processor = TextProcessor()
        self._initialize_store()

    def _initialize_store(self):
        """Initialize vector store components."""
        self.enabled = True
        self.use_embeddings = False
        self.model = None
        self.index = None
        
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
        if not hasattr(self, 'embeddings_initialized'):
            self.embeddings_initialized = False
        
        # Initialize embeddings if available
        if EMBEDDINGS_AVAILABLE:
            self._initialize_embeddings()
        
        # Load existing store
        self._load_existing_store()

    def _initialize_embeddings(self):
        """Initialize embeddings model with error handling."""
        if not EMBEDDINGS_AVAILABLE or self.embeddings_initialized:
            return
        
        try:
            logger.info("Initializing sentence transformers model...")
            self.model = SentenceTransformer(self.model_name)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.use_embeddings = True
            self.embeddings_initialized = True
            logger.info("Embeddings initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            logger.info("Falling back to text-based search only.")
            self.use_embeddings = False
            self.embeddings_initialized = True

    async def add_document(self, file_id: str, filename: str, text: str) -> int:
        chunks = self.chunk_manager.chunk_text(text)
        
        if self.use_embeddings and self.model:
            try:
                embeddings = await self._compute_embeddings(chunks)
                self.index.add(embeddings)
            except Exception as e:
                logger.error(f"Error computing embeddings: {e}")
                self.use_embeddings = False
        
        start_idx = len(self.chunks)
        new_metadata = [
            {"file_id": file_id, "filename": filename, "chunk_id": start_idx + i, "chunk_text": chunk}
            for i, chunk in enumerate(chunks)
        ]
        self.chunks.extend(chunks)
        self.metadata.extend(new_metadata)
        self.documents.append({
            "file_id": file_id, "filename": filename,
            "chunk_count": len(chunks), "start_idx": start_idx
        })
        if self.use_embeddings:
            await self.save_store()
        return len(chunks)

    async def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch)
            faiss.normalize_L2(embeddings)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.metadata:
            return []
        if self.use_embeddings and self.model and self.index and self.index.ntotal > 0:
            try:
                vector_results = await self._vector_search(query, top_k)
                if vector_results:
                    return vector_results
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        return await self._text_search(query, top_k)

    async def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["similarity_score"] = float(score)
                results.append(result)
        return results

    def remove_document(self, file_id: str) -> bool:
        if not self.document_exists(file_id):
            return False

        self.documents = [d for d in self.documents if d["file_id"] != file_id]
        self.metadata = [m for m in self.metadata if m["file_id"] != file_id]
        self.chunks = [m["chunk_text"] for m in self.metadata]

        if self.use_embeddings and self.model and len(self.chunks) > 0:
            loop = __import__("asyncio").new_event_loop()
            try:
                embeddings = self.model.encode(self.chunks)
                faiss.normalize_L2(embeddings)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings)
            finally:
                loop.close()
        elif self.use_embeddings:
            self.index = faiss.IndexFlatIP(self.dimension)

        import threading
        threading.Thread(target=lambda: __import__("asyncio").run(self.save_store()), daemon=True).start()
        logger.info(f"Removed document {file_id} and rebuilt index.")
        return True

    def document_exists(self, file_id: str) -> bool:
        return any(d["file_id"] == file_id for d in self.documents)

    def search_by_document(self, file_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        doc_metadata = [m for m in self.metadata if m["file_id"] == file_id]
        if not doc_metadata:
            return []
        query_lower = query.lower()
        results = []
        for meta in doc_metadata:
            text = meta["chunk_text"].lower()
            score = sum(text.count(word) for word in query_lower.split()) / max(len(text.split()), 1)
            if score > 0:
                result = meta.copy()
                result["similarity_score"] = min(score, 1.0)
                results.append(result)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def update_document_filename(self, file_id: str, new_filename: str) -> bool:
        updated = False
        for doc in self.documents:
            if doc["file_id"] == file_id:
                doc["filename"] = new_filename
                updated = True
        for meta in self.metadata:
            if meta["file_id"] == file_id:
                meta["filename"] = new_filename
        return updated

    async def _text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_clean = self.text_processor.preprocess_text(query)
        query_tokens = self.text_processor.tokenize(query_clean)
        if not query_tokens:
            return []
        results = []
        for metadata in self.metadata:
            chunk_text = metadata["chunk_text"]
            chunk_clean = self.text_processor.preprocess_text(chunk_text)
            chunk_tokens = self.text_processor.tokenize(chunk_clean)
            if not chunk_tokens:
                continue
            combined_score = await self._calculate_similarity_scores(
                query, query_tokens, chunk_text, chunk_tokens)
            if combined_score > 0:
                result = metadata.copy()
                result["similarity_score"] = combined_score
                results.append(result)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    async def _calculate_similarity_scores(
        self, query: str, query_tokens: List[str],
        chunk_text: str, chunk_tokens: List[str]) -> float:
        """Calculate combined similarity score with optimized algorithms."""
        try:
            scores = {
                'tfidf': self._calculate_tfidf_similarity(query_tokens, chunk_tokens),
                'bm25': self._calculate_bm25_score(query_tokens, chunk_tokens),
                'jaccard': self._calculate_jaccard_similarity(query_tokens, chunk_tokens),
                'semantic': self._calculate_semantic_overlap(query, chunk_text),
                'ngram': self._calculate_ngram_similarity(query, chunk_text)
            }
            
            weights = {
                'tfidf': 0.3,
                'bm25': 0.25,
                'jaccard': 0.15,
                'semantic': 0.2,
                'ngram': 0.1
            }
            
            raw = sum(score * weights[metric] for metric, score in scores.items())
            return min(raw, 1.0)
        except Exception as e:
            logger.error(f"Error calculating similarity scores: {e}")
            return 0.0

    def _calculate_tfidf_similarity(self, query_tokens: List[str], chunk_tokens: List[str]) -> float:
        if not query_tokens or not chunk_tokens:
            return 0.0
        query_tf = Counter(query_tokens)
        chunk_tf = Counter(chunk_tokens)
        all_terms = set(query_tokens) | set(chunk_tokens)
        score = 0.0
        n = len(self.chunks)
        for term in all_terms:
            if term in query_tf and term in chunk_tf:
                tf_query = query_tf[term] / len(query_tokens)
                tf_chunk = chunk_tf[term] / len(chunk_tokens)
                doc_freq = sum(1 for c in self.chunks if term in c)
                idf = math.log((n + 1) / (doc_freq + 1))
                score += tf_query * tf_chunk * idf
        return score

    def _calculate_bm25_score(self, query_tokens: List[str], chunk_tokens: List[str]) -> float:
        if not query_tokens or not chunk_tokens:
            return 0.0
        k1 = 1.5
        b = 0.75
        avg_len = sum(len(c.split()) for c in self.chunks) / len(self.chunks) if self.chunks else 1
        n = len(self.chunks)
        score = 0.0
        chunk_tf = Counter(chunk_tokens)
        chunk_len = len(chunk_tokens)
        for term in query_tokens:
            if term in chunk_tf:
                tf = chunk_tf[term]
                doc_freq = sum(1 for c in self.chunks if term in c)
                idf = math.log((n + 1) / (doc_freq + 1))
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * chunk_len / avg_len)))
        return score

    def _calculate_jaccard_similarity(self, query_tokens: List[str], chunk_tokens: List[str]) -> float:
        """Calculate Jaccard similarity score."""
        if not query_tokens or not chunk_tokens:
            return 0.0
        
        query_set = set(query_tokens)
        chunk_set = set(chunk_tokens)
        
        intersection = len(query_set & chunk_set)
        union = len(query_set | chunk_set)
        
        return intersection / union if union > 0 else 0.0

    def _calculate_semantic_overlap(self, query: str, chunk_text: str) -> float:
        """Calculate semantic overlap using word embeddings."""
        if not query or not chunk_text:
            return 0.0
        
        try:
            query_words = set(query.lower().split())
            chunk_words = set(chunk_text.lower().split())
            
            overlap = len(query_words & chunk_words)
            total = len(query_words)
            
            return overlap / total if total > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_ngram_similarity(self, query: str, chunk_text: str, n: int = 3) -> float:
        """Calculate n-gram similarity score."""
        if not query or not chunk_text or len(query) < n or len(chunk_text) < n:
            return 0.0
        
        def get_ngrams(text: str) -> Set[str]:
            return {text[i:i+n] for i in range(len(text) - n + 1)}
        
        query_ngrams = get_ngrams(query.lower())
        chunk_ngrams = get_ngrams(chunk_text.lower())
        
        if not query_ngrams or not chunk_ngrams:
            return 0.0
        
        intersection = len(query_ngrams & chunk_ngrams)
        union = len(query_ngrams | chunk_ngrams)
        
        return intersection / union if union > 0 else 0.0

    async def save_store(self):
        try:
            store_dir = Path(self.vector_store_path)
            store_dir.mkdir(exist_ok=True)
            if self.use_embeddings and self.index:
                faiss.write_index(self.index, str(store_dir / "index.faiss"))
            store_data = {
                "documents": self.documents,
                "chunks": self.chunks,
                "metadata": self.metadata,
                "model_name": self.model_name,
                "dimension": self.dimension,
            }
            with open(store_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(store_data, f)
            logger.info("Vector store saved.")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise

    def _load_existing_store(self):
        try:
            store_dir = Path(self.vector_store_path)
            json_path = store_dir / "metadata.json"
            pkl_path = store_dir / "metadata.pkl"
            index_path = store_dir / "index.faiss"

            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    store_data = json.load(f)
            elif pkl_path.exists():
                import pickle as _pickle
                with open(pkl_path, "rb") as f:
                    store_data = _pickle.load(f)
                pkl_path.unlink()
            else:
                return

            self.documents = store_data.get("documents", [])
            self.chunks = store_data.get("chunks", [])
            self.metadata = store_data.get("metadata", [])

            if self.use_embeddings and index_path.exists():
                self.index = faiss.read_index(str(index_path))

            logger.info(f"Loaded vector store: {len(self.documents)} docs, {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.documents = []
            self.chunks = []
            self.metadata = []