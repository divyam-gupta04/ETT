"""
Embeddings Module
Generates vector embeddings using sentence-transformers (runs locally on CPU)
"""

import sys
from pathlib import Path
from typing import List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    Uses HuggingFace models that run locally on CPU.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            print(f"[*] Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            print(f"[OK] Embedding model loaded successfully")
        return self._model
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        print(f"[*] Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        print(f"[OK] Generated {len(embeddings)} embeddings")
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()


# Singleton instance
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get the singleton embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def embed_query(text: str) -> List[float]:
    """Convenience function to embed a query."""
    return get_embedding_model().embed_query(text)


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Convenience function to embed documents."""
    return get_embedding_model().embed_documents(texts)


if __name__ == "__main__":
    # Test embeddings
    model = EmbeddingModel()
    
    test_texts = [
        "What is machine learning?",
        "Deep learning is a subset of machine learning.",
        "Python is a programming language."
    ]
    
    embeddings = model.embed_documents(test_texts)
    print(f"\nEmbedding dimension: {model.get_embedding_dimension()}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"First embedding (truncated): {embeddings[0][:5]}...")
