"""
Orchestrate the full document processing pipeline:
Load → Chunk → Embed
"""

from typing import List, Optional
from langchain_classic.schema import Document

from src.ingestion.document_loader import DocumentLoader
from src.embedding.chunking import DocumentChunker
from src.embedding.embedder import DocumentEmbedder
from src.embedding.chunk_saver import save_processed_chunks
from src.constants import ChunkingStrategy, CHUNK_STRATEGY_NONE, DATA_SOURCE_ARXIV
from src.utils.logging_config import setup_logging
from pathlib import Path

logger = setup_logging("document_processor", log_dir=Path("logs") / "embedding")


class DocumentProcessor:
    """
    Complete pipeline: Load docs → Chunk → Embed
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_strategy: str | ChunkingStrategy = ChunkingStrategy.NONE,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the complete processing pipeline.
        """
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy
        )
        self.embedder = DocumentEmbedder(model_name=embedding_model)
        
        logger.info("Initialized DocumentProcessor pipeline")
    
    def process_documents(
        self,
        include_full_text: bool = False,
        max_documents: Optional[int] = None,
        batch_size: int = 32,
        save_to_disk: bool = True
    ) -> List[Document]:
        """
        Run complete pipeline: Load → Chunk → Embed → Save
        
        Args:
            include_full_text: Include PDF text or just abstracts
            max_documents: Limit number of documents (for testing)
            batch_size: Batch size for embedding generation
            save_to_disk: Whether to save processed chunks to data/processed/
            
        Returns:
            List of Document objects with embeddings in metadata
        """
        logger.info("Starting document processing pipeline")
        
        # Step 1: Load documents
        documents = self.loader.load_all_documents(
            include_full_text=include_full_text,
            max_documents=max_documents,
            save_to_disk=save_to_disk
        )
        
        if not documents:
            error_msg = (
                "No documents were loaded. This usually means no metadata files exist. "
                f"Expected metadata directory: {self.loader.metadata_dir}. "
                "Please ensure you have fetched and saved ArXiv metadata first."
            )
            logger.warning(error_msg)
            return []
        
        # Step 2: Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        # Step 3: Generate embeddings
        embedded_chunks = self.embedder.embed_documents(
            chunks,
            batch_size=batch_size
        )
        
        # Step 4: Save processed chunks to disk (per architecture Rule 1)
        if save_to_disk and embedded_chunks:
            suffix = "full_text" if include_full_text else "abstracts"
            try:
                saved_path = save_processed_chunks(
                    embedded_chunks,
                    source=DATA_SOURCE_ARXIV,
                    suffix=suffix
                )
                logger.info(f"Saved processed chunks to {saved_path}")
            except Exception as exc:
                logger.warning(
                    "Failed to save processed chunks (continuing anyway)",
                    exc_info=exc
                )
        
        logger.info(
            "Document processing pipeline completed",
            extra={
                "input_documents": len(documents),
                "output_chunks": len(embedded_chunks),
                "saved_to_disk": save_to_disk
            }
        )
        
        return embedded_chunks


def process_arxiv_abstracts(max_documents: Optional[int] = None) -> List[Document]:
    """
    Process ArXiv papers (abstracts only, no chunking).
    
    Args:
        max_documents: Limit for testing (e.g., 100)
        
    Returns:
        List of embedded documents ready for vector store
    """
    processor = DocumentProcessor(
        embedding_model="all-MiniLM-L6-v2",
        chunk_strategy=CHUNK_STRATEGY_NONE  # No chunking for abstracts
    )
    
    return processor.process_documents(
        include_full_text=False,  # Abstracts only
        max_documents=max_documents
    )