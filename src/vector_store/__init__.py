"""
Vector store module for managing document embeddings.
"""

from src.vector_store.chroma_store import (
    ChromaVectorStore,
    create_vector_store_from_processed_chunks,
)

__all__ = [
    "ChromaVectorStore",
    "create_vector_store_from_processed_chunks",
]


