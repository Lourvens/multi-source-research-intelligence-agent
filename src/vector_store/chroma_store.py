"""
ChromaDB vector store implementation using LangChain wrapper.

This module provides a vector store manager for storing and retrieving
document embeddings using ChromaDB with LangChain integration.
"""

from typing import List, Optional
from pathlib import Path

from langchain_classic.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from src import config
from src.utils.logging_config import setup_logging
from src.constants import DATA_SOURCE_ARXIV

logger = setup_logging("chroma_store", log_dir=Path("logs") / "vector_store")


class ChromaVectorStore:
    """
    ChromaDB vector store manager using LangChain wrapper.
    
    Provides methods to:
    - Create vector store from documents
    - Add documents to existing store
    - Search for similar documents
    - Manage collections
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str | Path = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection.
                Defaults to config.COLLECTION_NAME
            persist_directory: Directory to persist the vector store.
                Defaults to config.VECTOR_DB_DIR
            embedding_model: Embedding model name for HuggingFace.
                Defaults to "all-MiniLM-L6-v2"
        """
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.persist_directory = Path(persist_directory or config.VECTOR_DB_DIR)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if not embedding_model.startswith("sentence-transformers/"):
            embedding_model = f"sentence-transformers/{embedding_model}"
        
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vector_store: Optional[Chroma] = None
        
        logger.info(
            "Initialized ChromaVectorStore",
            extra={
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory),
                "embedding_model": embedding_model
            }
        )
    
    def create_from_documents(
        self,
        documents: List[Document],
        collection_name: str = None
    ) -> Chroma:
        """
        Create a new ChromaDB vector store from a list of documents.
        
        Args:
            documents: List of Document objects with embeddings in metadata
            collection_name: Optional collection name (overrides default)
            
        Returns:
            Chroma vector store instance
        """
        if not documents:
            logger.warning("No documents provided to create vector store")
            return None
        
        collection_name = collection_name or self.collection_name
        
        logger.info(
            "Creating vector store from documents",
            extra={
                "num_documents": len(documents),
                "collection_name": collection_name
            }
        )
        
        # Filter complex metadata (lists, dicts) that ChromaDB doesn't support
        # ChromaDB only supports: str, int, float, bool, None
        def filter_metadata(metadata: dict) -> dict:
            """Filter metadata to only include ChromaDB-compatible types."""
            filtered = {}
            for key, value in metadata.items():
                # Skip embedding as it's not needed in metadata (Chroma generates it)
                if key == "embedding":
                    continue
                # Only include simple types
                if isinstance(value, (str, int, float, bool, type(None))):
                    filtered[key] = value
                # Convert lists to comma-separated strings
                elif isinstance(value, list):
                    filtered[key] = ", ".join(str(v) for v in value)
                # Skip other complex types
            return filtered
        
        filtered_documents = []
        for doc in documents:
            filtered_metadata = filter_metadata(doc.metadata)
            filtered_doc = Document(
                page_content=doc.page_content,
                metadata=filtered_metadata
            )
            filtered_documents.append(filtered_doc)
        
        try:
            self.vector_store = Chroma.from_documents(
                documents=filtered_documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=str(self.persist_directory)
            )
            
            logger.info(
                "Successfully created vector store",
                extra={
                    "collection_name": collection_name,
                    "num_documents": len(documents),
                    "persist_directory": str(self.persist_directory)
                }
            )
            
            return self.vector_store
            
        except Exception as exc:
            logger.error(
                "Failed to create vector store",
                extra={"collection_name": collection_name},
                exc_info=exc
            )
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        collection_name: str = None
    ) -> List[str]:
        """
        Add documents to an existing vector store.
        
        Args:
            documents: List of Document objects to add
            collection_name: Optional collection name (overrides default)
            
        Returns:
            List of document IDs added to the store
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []
        
        collection_name = collection_name or self.collection_name
        
        # Load existing vector store if not already loaded
        if self.vector_store is None:
            self._load_vector_store(collection_name)
        
        # Filter complex metadata before adding
        def filter_metadata(metadata: dict) -> dict:
            """Filter metadata to only include ChromaDB-compatible types."""
            filtered = {}
            for key, value in metadata.items():
                if key == "embedding":
                    continue
                if isinstance(value, (str, int, float, bool, type(None))):
                    filtered[key] = value
                elif isinstance(value, list):
                    filtered[key] = ", ".join(str(v) for v in value)
            return filtered
        
        filtered_documents = []
        for doc in documents:
            filtered_metadata = filter_metadata(doc.metadata)
            filtered_doc = Document(
                page_content=doc.page_content,
                metadata=filtered_metadata
            )
            filtered_documents.append(filtered_doc)
        
        try:
            ids = self.vector_store.add_documents(filtered_documents)
            
            logger.info(
                "Added documents to vector store",
                extra={
                    "num_documents": len(documents),
                    "collection_name": collection_name
                }
            )
            
            return ids
            
        except Exception as exc:
            logger.error(
                "Failed to add documents to vector store",
                extra={"collection_name": collection_name},
                exc_info=exc
            )
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        collection_name: str = None
    ) -> List[Document]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Query text to search for
            k: Number of similar documents to return
            collection_name: Optional collection name (overrides default)
            
        Returns:
            List of most similar Document objects
        """
        collection_name = collection_name or self.collection_name
        
        # Load existing vector store if not already loaded
        if self.vector_store is None:
            self._load_vector_store(collection_name)
        
        if self.vector_store is None:
            logger.error("Vector store not initialized. Create it first.")
            return []
        
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            logger.info(
                "Performed similarity search",
                extra={
                    "query": query[:100],  # Log first 100 chars
                    "k": k,
                    "results_count": len(results),
                    "collection_name": collection_name
                }
            )
            
            return results
            
        except Exception as exc:
            logger.error(
                "Failed to perform similarity search",
                extra={"query": query[:100], "k": k},
                exc_info=exc
            )
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        collection_name: str = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text to search for
            k: Number of similar documents to return
            collection_name: Optional collection name (overrides default)
            
        Returns:
            List of tuples (Document, score) sorted by similarity
        """
        collection_name = collection_name or self.collection_name
        
        # Load existing vector store if not already loaded
        if self.vector_store is None:
            self._load_vector_store(collection_name)
        
        if self.vector_store is None:
            logger.error("Vector store not initialized. Create it first.")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(
                "Performed similarity search with scores",
                extra={
                    "query": query[:100],
                    "k": k,
                    "results_count": len(results),
                    "collection_name": collection_name
                }
            )
            
            return results
            
        except Exception as exc:
            logger.error(
                "Failed to perform similarity search with scores",
                extra={"query": query[:100], "k": k},
                exc_info=exc
            )
            raise
    
    def _load_vector_store(self, collection_name: str = None) -> None:
        """
        Load an existing vector store from disk.
        
        Args:
            collection_name: Optional collection name (overrides default)
        """
        collection_name = collection_name or self.collection_name
        
        try:
            self.vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            
            logger.info(
                "Loaded existing vector store",
                extra={
                    "collection_name": collection_name,
                    "persist_directory": str(self.persist_directory)
                }
            )
            
        except Exception as exc:
            logger.warning(
                "Failed to load existing vector store (may not exist yet)",
                extra={"collection_name": collection_name},
                exc_info=exc
            )
            self.vector_store = None
    
    def get_collection_info(self) -> dict:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection metadata
        """
        # Try to load if not already loaded
        if self.vector_store is None:
            self._load_vector_store()
        
        if self.vector_store is None:
            return {
                "collection_name": self.collection_name,
                "exists": False,
                "count": 0
            }
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # If count is 0, collection might not really exist (empty collection)
            return {
                "collection_name": self.collection_name,
                "exists": count > 0,  # Only exists if it has documents
                "count": count,
                "persist_directory": str(self.persist_directory),
                "embedding_model": self.embedding_model
            }
            
        except Exception as exc:
            logger.error(
                "Failed to get collection info",
                exc_info=exc
            )
            return {
                "collection_name": self.collection_name,
                "exists": False,
                "error": str(exc)
            }
    
    def delete_collection(self, collection_name: str = None) -> bool:
        """
        Delete a collection from the vector store.
        
        Args:
            collection_name: Optional collection name (overrides default)
            
        Returns:
            True if deletion was successful
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # Load the collection first
            if self.vector_store is None:
                self._load_vector_store(collection_name)
            
            if self.vector_store is None:
                logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            # Delete the collection
            self.vector_store.delete_collection()
            self.vector_store = None
            
            logger.info(
                "Deleted collection",
                extra={"collection_name": collection_name}
            )
            
            return True
            
        except Exception as exc:
            logger.error(
                "Failed to delete collection",
                extra={"collection_name": collection_name},
                exc_info=exc
            )
            return False


def create_vector_store_from_processed_chunks(
    chunks: List[Document],
    collection_name: str = None,
    persist_directory: str | Path = None
) -> ChromaVectorStore:
    """
    Convenience function to create a vector store from processed chunks.
    
    Args:
        chunks: List of Document objects with embeddings
        collection_name: Optional collection name
        persist_directory: Optional persist directory
        
    Returns:
        ChromaVectorStore instance
    """
    store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    store.create_from_documents(chunks, collection_name=collection_name)
    return store

