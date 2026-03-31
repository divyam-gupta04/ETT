"""
Vector Store Module
ChromaDB-based persistent vector database for document storage and retrieval
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from src.embeddings import get_embedding_model


class VectorStore:
    """
    ChromaDB-based vector store for document storage and semantic search.
    Provides persistent storage that survives restarts.
    """
    
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(CHROMA_DIR)
        self.embedding_model = get_embedding_model()
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            
            print(f"[*] Initializing ChromaDB at: {self.persist_directory}")
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            print(f"[OK] ChromaDB initialized")
        return self._client
    
    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"[*] Collection '{self.collection_name}' ready ({self._collection.count()} documents)")
        return self._collection
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            Number of documents added
        """
        if not documents:
            print("[WARNING] No documents to add")
            return 0
        
        # Prepare data for ChromaDB
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Generate unique IDs
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Add to ChromaDB
        print(f"[*] Adding {len(documents)} documents to vector store...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"[OK] Added {len(documents)} documents successfully")
        
        return len(documents)
    
    def query(
        self,
        query_text: str,
        top_k: int = TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with document, metadata, and similarity score
        """
        if self.collection.count() == 0:
            print("[WARNING] Vector store is empty. Please ingest documents first.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query_text)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity score (1 - distance for cosine)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance
                
                formatted_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": similarity,
                    "id": results["ids"][0][i] if results["ids"] else f"doc_{i}"
                })
        
        print(f"[*] Found {len(formatted_results)} relevant documents")
        return formatted_results
    
    def clear(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = None
            print(f"[*] Cleared collection '{self.collection_name}'")
            return True
        except Exception as e:
            print(f"[ERROR] Error clearing collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_directory
        }


# Singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get the singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


if __name__ == "__main__":
    # Test vector store
    store = VectorStore()
    
    # Test documents
    test_docs = [
        Document(page_content="Machine learning is a type of AI.", metadata={"source": "test.txt"}),
        Document(page_content="Python is great for data science.", metadata={"source": "test.txt"}),
    ]
    
    store.add_documents(test_docs)
    
    results = store.query("What is machine learning?")
    for r in results:
        print(f"\nScore: {r['similarity']:.3f}")
        print(f"Content: {r['content'][:100]}...")
        print(f"Source: {r['metadata'].get('source', 'Unknown')}")
