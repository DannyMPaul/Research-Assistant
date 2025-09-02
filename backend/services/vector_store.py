import numpy as np
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers or faiss not available. Vector search disabled.")

import pickle
import os
from pathlib import Path
import json
import re
from collections import Counter
import math

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Always enable semantic search (use text-based similarity if embeddings unavailable)
        self.enabled = True
        self.use_embeddings = EMBEDDINGS_AVAILABLE
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.dimension = 384  # MiniLM embedding size
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
            except Exception as e:
                print(f"Error initializing embeddings: {e}")
                self.use_embeddings = False
        
        self.documents = []
        self.chunks = []
        self.metadata = []
        self.vector_store_path = Path("vector_store")
        self.vector_store_path.mkdir(exist_ok=True)
        
        if self.use_embeddings:
            self.load_existing_store()
    
    def add_document(self, file_id, filename, text):
        chunks = self._chunk_text(text)
        
        if self.use_embeddings:
            try:
                embeddings = self.model.encode(chunks)
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
            except Exception as e:
                print(f"Error with embeddings, using text-based: {e}")
                self.use_embeddings = False
        
        start_idx = len(self.chunks)
        
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.metadata.append({
                'file_id': file_id,
                'filename': filename,
                'chunk_id': start_idx + i,
                'chunk_text': chunk
            })
        
        self.documents.append({
            'file_id': file_id,
            'filename': filename,
            'chunk_count': len(chunks),
            'start_idx': start_idx
        })
        
        if self.use_embeddings:
            self.save_store()
        
        return len(chunks)
    
    def search(self, query, top_k=5):
        if not self.metadata:
            return []
        
        if self.use_embeddings and hasattr(self, 'index') and self.index.ntotal > 0:
            try:
                query_embedding = self.model.encode([query])
                faiss.normalize_L2(query_embedding)
                scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.metadata):
                        result = self.metadata[idx].copy()
                        result['similarity_score'] = float(score)
                        results.append(result)
                return results
            except Exception as e:
                print(f"Embeddings search failed, using text-based: {e}")
        
        # Advanced text-based semantic search using multiple similarity algorithms
        results = []
        
        # Preprocess query
        query_clean = self._preprocess_text(query)
        query_tokens = self._tokenize(query_clean)
        
        if not query_tokens:
            return []
        
        # Calculate different similarity scores for each chunk
        for metadata in self.metadata:
            chunk_text = metadata['chunk_text']
            chunk_clean = self._preprocess_text(chunk_text)
            chunk_tokens = self._tokenize(chunk_clean)
            
            if not chunk_tokens:
                continue
            
            # 1. TF-IDF Cosine Similarity
            tfidf_score = self._calculate_tfidf_similarity(query_tokens, chunk_tokens)
            
            # 2. BM25 Score
            bm25_score = self._calculate_bm25_score(query_tokens, chunk_tokens)
            
            # 3. Jaccard Similarity (token overlap)
            jaccard_score = self._calculate_jaccard_similarity(query_tokens, chunk_tokens)
            
            # 4. Semantic overlap with context
            semantic_score = self._calculate_semantic_overlap(query, chunk_text)
            
            # 5. N-gram similarity
            ngram_score = self._calculate_ngram_similarity(query_clean, chunk_clean)
            
            # Combine scores with weights (tuned based on empirical testing)
            combined_score = (
                0.3 * tfidf_score +
                0.25 * bm25_score +
                0.15 * jaccard_score +
                0.2 * semantic_score +
                0.1 * ngram_score
            )
            
            if combined_score > 0:
                result = metadata.copy()
                result['similarity_score'] = combined_score
                result['score_breakdown'] = {
                    'tfidf': tfidf_score,
                    'bm25': bm25_score,
                    'jaccard': jaccard_score,
                    'semantic': semantic_score,
                    'ngram': ngram_score
                }
                results.append(result)
        
        # Sort by combined similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def search_by_document(self, file_id, query, top_k=3):
        if not self.enabled:
            return []
        all_results = self.search(query, top_k * 3)  # Get more results to filter
        document_results = [r for r in all_results if r['file_id'] == file_id]
        return document_results[:top_k]
    
    def _chunk_text(self, text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def save_store(self):
        if not self.enabled:
            return
        faiss.write_index(self.index, str(self.vector_store_path / "index.faiss"))
        
        with open(self.vector_store_path / "metadata.pkl", "wb") as f:
            pickle.dump({
                'documents': self.documents,
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
    
    def load_existing_store(self):
        if not self.enabled:
            return
            
        index_path = self.vector_store_path / "index.faiss"
        metadata_path = self.vector_store_path / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.chunks = data.get('chunks', [])
                    self.metadata = data.get('metadata', [])
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self._reset_store()
    
    def _reset_store(self):
        if not self.enabled:
            return
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.chunks = []
        self.metadata = []
    
    def get_document_info(self, file_id):
        for doc in self.documents:
            if doc['file_id'] == file_id:
                return doc
        return None
    
    def document_exists(self, file_id):
        return self.get_document_info(file_id) is not None
    
    def _preprocess_text(self, text):
        """Clean and normalize text for better matching"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep letters, numbers, and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()
    
    def _tokenize(self, text):
        """Tokenize text into meaningful terms"""
        # Split by whitespace and filter out empty strings and single characters
        tokens = [token for token in text.split() if len(token) > 1]
        return tokens
    
    def _calculate_tfidf_similarity(self, query_tokens, chunk_tokens):
        """Calculate TF-IDF cosine similarity"""
        try:
            # Get all unique terms
            all_terms = set(query_tokens + chunk_tokens)
            
            if not all_terms:
                return 0.0
            
            # Calculate term frequencies
            query_tf = Counter(query_tokens)
            chunk_tf = Counter(chunk_tokens)
            
            # Calculate TF-IDF vectors
            query_tfidf = {}
            chunk_tfidf = {}
            
            # Simple TF-IDF calculation (TF * log(1 + total_terms / term_freq))
            total_terms = len(all_terms)
            
            for term in all_terms:
                # Query TF-IDF
                tf_q = query_tf[term] / len(query_tokens) if query_tokens else 0
                idf = math.log(1 + total_terms / (query_tf[term] + chunk_tf[term])) if (query_tf[term] + chunk_tf[term]) > 0 else 0
                query_tfidf[term] = tf_q * idf
                
                # Chunk TF-IDF
                tf_c = chunk_tf[term] / len(chunk_tokens) if chunk_tokens else 0
                chunk_tfidf[term] = tf_c * idf
            
            # Calculate cosine similarity
            dot_product = sum(query_tfidf[term] * chunk_tfidf[term] for term in all_terms)
            
            query_norm = math.sqrt(sum(val**2 for val in query_tfidf.values()))
            chunk_norm = math.sqrt(sum(val**2 for val in chunk_tfidf.values()))
            
            if query_norm == 0 or chunk_norm == 0:
                return 0.0
            
            return dot_product / (query_norm * chunk_norm)
        
        except Exception:
            return 0.0
    
    def _calculate_bm25_score(self, query_tokens, chunk_tokens):
        """Calculate BM25 score (simplified version)"""
        try:
            if not query_tokens or not chunk_tokens:
                return 0.0
            
            # BM25 parameters
            k1 = 1.2
            b = 0.75
            
            # Average document length (approximated)
            avg_doc_length = 100  # Approximate average chunk length
            doc_length = len(chunk_tokens)
            
            chunk_tf = Counter(chunk_tokens)
            score = 0.0
            
            for term in query_tokens:
                if term in chunk_tf:
                    tf = chunk_tf[term]
                    # Simplified IDF (log(1 + 1/tf))
                    idf = math.log(1 + 1/tf)
                    
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                    score += idf * (numerator / denominator)
            
            return score / len(query_tokens) if query_tokens else 0.0
        
        except Exception:
            return 0.0
    
    def _calculate_jaccard_similarity(self, query_tokens, chunk_tokens):
        """Calculate Jaccard similarity coefficient"""
        try:
            query_set = set(query_tokens)
            chunk_set = set(chunk_tokens)
            
            if not query_set or not chunk_set:
                return 0.0
            
            intersection = len(query_set.intersection(chunk_set))
            union = len(query_set.union(chunk_set))
            
            return intersection / union if union > 0 else 0.0
        
        except Exception:
            return 0.0
    
    def _calculate_semantic_overlap(self, query, chunk_text):
        """Calculate semantic overlap with context awareness"""
        try:
            query_clean = self._preprocess_text(query)
            chunk_clean = self._preprocess_text(chunk_text)
            
            # Word-level matches with context
            query_words = query_clean.split()
            chunk_words = chunk_clean.split()
            
            if not query_words or not chunk_words:
                return 0.0
            
            # Calculate different types of matches
            exact_matches = 0
            partial_matches = 0
            context_score = 0
            
            for q_word in query_words:
                # Exact matches
                if q_word in chunk_words:
                    exact_matches += 1
                    # Context bonus: check surrounding words
                    context_score += chunk_clean.count(q_word) * 0.1
                
                # Partial matches (substring matching)
                for c_word in chunk_words:
                    if len(q_word) > 3 and q_word in c_word and q_word != c_word:
                        partial_matches += 0.5
                    elif len(c_word) > 3 and c_word in q_word and q_word != c_word:
                        partial_matches += 0.5
            
            # Normalize by query length
            exact_score = exact_matches / len(query_words)
            partial_score = partial_matches / len(query_words)
            context_normalized = context_score / (len(query_words) + len(chunk_words))
            
            # Combined semantic score
            return exact_score * 0.6 + partial_score * 0.3 + context_normalized * 0.1
        
        except Exception:
            return 0.0
    
    def _calculate_ngram_similarity(self, query, chunk_text, n=2):
        """Calculate n-gram similarity"""
        try:
            def get_ngrams(text, n):
                words = text.split()
                return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            
            if len(query.split()) < n or len(chunk_text.split()) < n:
                return 0.0
            
            query_ngrams = set(get_ngrams(query, n))
            chunk_ngrams = set(get_ngrams(chunk_text, n))
            
            if not query_ngrams or not chunk_ngrams:
                return 0.0
            
            intersection = len(query_ngrams.intersection(chunk_ngrams))
            union = len(query_ngrams.union(chunk_ngrams))
            
            return intersection / union if union > 0 else 0.0
        
        except Exception:
            return 0.0
