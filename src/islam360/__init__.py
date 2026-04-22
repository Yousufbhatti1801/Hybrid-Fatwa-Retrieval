"""Islam360-only fatwa RAG pipeline (1536-d embeddings, dedicated Pinecone index)."""

from src.islam360.retrieve import Islam360Retriever, NO_RELEVANT_FATWA

__all__ = ["Islam360Retriever", "NO_RELEVANT_FATWA"]
