"""
Evaluation Metrics Module
Metrics for evaluating RAG system performance
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_retrieval_precision(
    retrieved_docs: List[Dict[str, Any]],
    relevant_sources: List[str]
) -> float:
    """
    Calculate retrieval precision.
    
    Precision = (Relevant Retrieved) / (Total Retrieved)
    
    Args:
        retrieved_docs: List of retrieved documents with metadata
        relevant_sources: List of source filenames that are considered relevant
        
    Returns:
        Precision score (0-1)
    """
    if not retrieved_docs:
        return 0.0
    
    relevant_retrieved = sum(
        1 for doc in retrieved_docs
        if doc.get("metadata", {}).get("source", "") in relevant_sources
    )
    
    return relevant_retrieved / len(retrieved_docs)


def calculate_retrieval_recall(
    retrieved_docs: List[Dict[str, Any]],
    relevant_sources: List[str],
    total_relevant_chunks: int
) -> float:
    """
    Calculate retrieval recall.
    
    Recall = (Relevant Retrieved) / (Total Relevant)
    
    Args:
        retrieved_docs: List of retrieved documents
        relevant_sources: List of relevant source filenames
        total_relevant_chunks: Total number of relevant chunks in corpus
        
    Returns:
        Recall score (0-1)
    """
    if total_relevant_chunks == 0:
        return 0.0
    
    relevant_retrieved = sum(
        1 for doc in retrieved_docs
        if doc.get("metadata", {}).get("source", "") in relevant_sources
    )
    
    return relevant_retrieved / total_relevant_chunks


def calculate_mrr(
    retrieved_docs: List[Dict[str, Any]],
    relevant_sources: List[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / (rank of first relevant document)
    
    Args:
        retrieved_docs: List of retrieved documents in ranked order
        relevant_sources: List of relevant source filenames
        
    Returns:
        MRR score (0-1)
    """
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("metadata", {}).get("source", "")
        if source in relevant_sources:
            return 1.0 / i
    
    return 0.0


def calculate_average_similarity(retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    Calculate average similarity score of retrieved documents.
    
    Args:
        retrieved_docs: List of retrieved documents with similarity scores
        
    Returns:
        Average similarity score
    """
    if not retrieved_docs:
        return 0.0
    
    similarities = [doc.get("similarity", 0) for doc in retrieved_docs]
    return sum(similarities) / len(similarities)


def evaluate_retrieval(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    relevant_sources: Optional[List[str]] = None,
    total_relevant_chunks: int = 0
) -> Dict[str, Any]:
    """
    Comprehensive retrieval evaluation.
    
    Args:
        query: The search query
        retrieved_docs: List of retrieved documents
        relevant_sources: Optional list of relevant sources (for precision/recall)
        total_relevant_chunks: Total relevant chunks (for recall)
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "query": query,
        "num_retrieved": len(retrieved_docs),
        "avg_similarity": calculate_average_similarity(retrieved_docs),
        "sources": list(set(
            doc.get("metadata", {}).get("source", "Unknown")
            for doc in retrieved_docs
        ))
    }
    
    # If ground truth is provided, calculate precision/recall
    if relevant_sources:
        metrics["precision"] = calculate_retrieval_precision(
            retrieved_docs, relevant_sources
        )
        metrics["mrr"] = calculate_mrr(retrieved_docs, relevant_sources)
        
        if total_relevant_chunks > 0:
            metrics["recall"] = calculate_retrieval_recall(
                retrieved_docs, relevant_sources, total_relevant_chunks
            )
    
    return metrics


def evaluate_answer_faithfulness(
    answer: str,
    context: str
) -> Dict[str, Any]:
    """
    Basic evaluation of answer faithfulness to context.
    
    This is a simple heuristic-based evaluation. For production,
    consider using LLM-based evaluation (e.g., RAGAS).
    
    Args:
        answer: Generated answer
        context: Retrieved context
        
    Returns:
        Faithfulness metrics
    """
    # Simple heuristics
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    
    # Calculate word overlap
    overlap = answer_words.intersection(context_words)
    overlap_ratio = len(overlap) / len(answer_words) if answer_words else 0
    
    # Check for citation presence
    has_citations = any(f"[{i}]" in answer for i in range(1, 10))
    
    # Check for hedging language (good for grounded responses)
    hedging_phrases = [
        "according to", "based on", "the document", "it states",
        "mentioned", "described", "indicates"
    ]
    has_hedging = any(phrase in answer.lower() for phrase in hedging_phrases)
    
    return {
        "word_overlap_ratio": overlap_ratio,
        "has_citations": has_citations,
        "has_hedging_language": has_hedging,
        "answer_length": len(answer),
        "context_length": len(context)
    }


def run_evaluation_suite(
    pipeline,
    test_queries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Run evaluation on a suite of test queries.
    
    Args:
        pipeline: RAG pipeline instance
        test_queries: List of test queries with optional ground truth
            Format: [{"query": "...", "relevant_sources": ["..."]}, ...]
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for test in test_queries:
        query = test["query"]
        relevant_sources = test.get("relevant_sources", [])
        
        # Run query
        response = pipeline.query(query)
        
        # Evaluate
        retrieval_metrics = evaluate_retrieval(
            query=query,
            retrieved_docs=[],  # Would need to expose this from pipeline
            relevant_sources=relevant_sources
        )
        
        faithfulness = evaluate_answer_faithfulness(
            answer=response.get("answer", ""),
            context=""  # Would need context from pipeline
        )
        
        results.append({
            "query": query,
            "answer": response.get("answer", "")[:200] + "...",
            "retrieval": retrieval_metrics,
            "faithfulness": faithfulness,
            "citations": len(response.get("citations", []))
        })
    
    return results


def print_evaluation_report(results: List[Dict[str, Any]]):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Query {i} ---")
        print(f"Q: {result['query'][:50]}...")
        print(f"A: {result['answer'][:100]}...")
        print(f"Citations: {result['citations']}")
        print(f"Retrieval metrics: {result['retrieval']}")
        print(f"Faithfulness: {result['faithfulness']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    test_docs = [
        {"content": "ML is a type of AI", "metadata": {"source": "ml.pdf"}, "similarity": 0.85},
        {"content": "Python is great", "metadata": {"source": "python.pdf"}, "similarity": 0.72},
    ]
    
    metrics = evaluate_retrieval(
        query="What is machine learning?",
        retrieved_docs=test_docs,
        relevant_sources=["ml.pdf"]
    )
    
    print("Retrieval Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
