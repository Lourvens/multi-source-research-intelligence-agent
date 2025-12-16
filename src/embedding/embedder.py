"""
Generate embeddings for documents using various embedding models.
"""

from typing import List
from langchain_classic.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.utils.logging_config import setup_logging
from pathlib import Path

logger = setup_logging("embedder", log_dir=Path("logs") / "embedding")


class DocumentEmbedder:
    """
    Generate embeddings for documents using HuggingFace models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a specific model.
        
        Args:
            model_name: Name of the HuggingFace embedding model.
                Default: "all-MiniLM-L6-v2" (sentence-transformers)
        """
        # Construct full model path for sentence-transformers
        if not model_name.startswith("sentence-transformers/"):
            model_name = f"sentence-transformers/{model_name}"
        
        self.model_name = model_name
        
        logger.info(
            "Initializing DocumentEmbedder",
            extra={"model_name": model_name}
        )
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},  # Use CPU by default
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("DocumentEmbedder initialized successfully")
        except Exception as exc:
            logger.error(
                "Failed to initialize DocumentEmbedder",
                extra={"model_name": model_name},
                exc_info=exc
            )
            raise
    
    def embed_documents(
        self,
        documents: List[Document],
        batch_size: int = 32
    ) -> List[Document]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects to embed
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of Document objects with embeddings added to metadata
            under the key "embedding"
        """
        if not documents:
            logger.warning("No documents to embed")
            return []
        
        logger.info(
            "Starting document embedding",
            extra={
                "num_documents": len(documents),
                "batch_size": batch_size,
                "model": self.model_name
            }
        )
        
        # Extract texts for embedding
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(
                    f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} documents"
                )
        
        # Add embeddings to document metadata
        embedded_documents = []
        for doc, embedding in zip(documents, all_embeddings):
            # Create new document with embedding in metadata
            new_metadata = doc.metadata.copy()
            new_metadata["embedding"] = embedding
            new_metadata["embedding_model"] = self.model_name
            
            embedded_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata
            )
            embedded_documents.append(embedded_doc)
        
        logger.info(
            "Completed document embedding",
            extra={
                "num_documents": len(embedded_documents),
                "embedding_dim": len(all_embeddings[0]) if all_embeddings else 0
            }
        )
        
        return embedded_documents
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self.embeddings.embed_query(query)
