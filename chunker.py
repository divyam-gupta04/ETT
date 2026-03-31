"""
Text Chunker Module
Splits documents into smaller chunks for embedding and retrieval
"""

import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS


def create_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = SEPARATORS
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with configurable settings.
    
    Args:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to use for splitting (in order of priority)
        
    Returns:
        Configured RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into smaller chunks while preserving metadata.
    
    Args:
        documents: List of Document objects to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    if not documents:
        return []
    
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    chunked_documents = []
    
    for doc in documents:
        # Split the document
        chunks = text_splitter.split_text(doc.page_content)
        
        # Create new Document objects with preserved metadata
        for i, chunk in enumerate(chunks):
            # Copy original metadata and add chunk info
            metadata = doc.metadata.copy()
            metadata["chunk_index"] = i
            metadata["total_chunks"] = len(chunks)
            
            chunked_doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            chunked_documents.append(chunked_doc)
    
    print(f"[*] Created {len(chunked_documents)} chunks from {len(documents)} documents")
    print(f"   Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars")
    
    return chunked_documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    metadata: dict = None
) -> List[Document]:
    """
    Chunk a raw text string into Document objects.
    
    Args:
        text: Raw text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        metadata: Optional metadata to attach to each chunk
        
    Returns:
        List of Document objects
    """
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_text(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc_metadata = metadata.copy() if metadata else {}
        doc_metadata["chunk_index"] = i
        doc_metadata["total_chunks"] = len(chunks)
        
        documents.append(Document(
            page_content=chunk,
            metadata=doc_metadata
        ))
    
    return documents


if __name__ == "__main__":
    # Test chunking
    from document_loader import load_documents
    from config import SAMPLE_DIR
    
    docs = load_documents(str(SAMPLE_DIR))
    if docs:
        chunks = chunk_documents(docs)
        print(f"\nSample chunk:")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
