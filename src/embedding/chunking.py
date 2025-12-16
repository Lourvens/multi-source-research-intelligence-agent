"""
Text chunking strategies for document processing.
"""

from typing import List, Dict, Optional
from langchain_classic.schema import Document
from langchain_classic.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from src.utils.logging_config import setup_logging
from src.constants import ChunkingStrategy, CHUNK_STRATEGY_NONE, CHUNK_STRATEGY_RECURSIVE, CHUNK_STRATEGY_TOKEN
from pathlib import Path

logger = setup_logging("chunking", log_dir=Path("logs") / "embedding")


class DocumentChunker:
    """
    Chunk documents using various strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str | ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ):
        """
        Args:
            chunk_size: Target size in tokens/characters
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy (ChunkingStrategy enum or string)
                - ChunkingStrategy.NONE: No chunking (for abstracts)
                - ChunkingStrategy.RECURSIVE: Recursive character splitting
                - ChunkingStrategy.TOKEN: Token-based splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Convert enum to string value if needed
        self.strategy = strategy.value if isinstance(strategy, ChunkingStrategy) else strategy
        
        self._init_splitter()
        
        logger.info(
            "Initialized DocumentChunker",
            extra={
                "strategy": strategy,
                "chunk_size": chunk_size,
                "overlap": chunk_overlap
            }
        )
    
    def _init_splitter(self):
        """Initialize the appropriate text splitter."""
        if self.strategy == CHUNK_STRATEGY_NONE:
            self.splitter = None
        
        elif self.strategy == CHUNK_STRATEGY_RECURSIVE:
            # Best for most documents - respects structure
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n\n",  # Section breaks
                    "\n\n",    # Paragraph breaks
                    "\n",      # Line breaks
                    ". ",      # Sentences
                    " ",       # Words
                    ""         # Characters
                ]
            )
        
        elif self.strategy == CHUNK_STRATEGY_TOKEN:
            # Token-based - more precise for embedding limits
            self.splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document.
        
        Args:
            document: LangChain Document to chunk
            
        Returns:
            List of Document chunks with enriched metadata
        """
        # Phase 1: No chunking for abstracts
        if self.strategy == CHUNK_STRATEGY_NONE:
            return [self._add_chunk_metadata(document, 0, 1)]
        
        # Chunk the document
        chunks = self.splitter.split_documents([document])
        
        # Enrich each chunk with metadata
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = self._add_chunk_metadata(
                chunk,
                chunk_index=i,
                total_chunks=len(chunks)
            )
            enriched_chunks.append(enriched_chunk)
        
        logger.debug(
            f"Chunked document",
            extra={
                "arxiv_id": document.metadata.get("arxiv_id"),
                "num_chunks": len(chunks)
            }
        )
        
        return enriched_chunks
    
    def _add_chunk_metadata(
        self,
        chunk: Document,
        chunk_index: int,
        total_chunks: int
    ) -> Document:
        """
        Add chunk-specific metadata to a document chunk.
        
        Important: Preserves all original metadata plus adds chunk info.
        """
        # Create unique chunk ID
        arxiv_id = chunk.metadata.get("arxiv_id", "unknown")
        chunk_id = f"{arxiv_id}_chunk_{chunk_index}"
        
        # Add chunk metadata
        chunk.metadata.update({
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_size": len(chunk.page_content),
        })
        
        return chunk
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of Documents to chunk
            
        Returns:
            Flattened list of all chunks from all documents
        """
        logger.info(f"Chunking {len(documents)} documents")
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(
            "Completed chunking",
            extra={
                "input_documents": len(documents),
                "output_chunks": len(all_chunks),
                "avg_chunks_per_doc": len(all_chunks) / len(documents)
            }
        )
        
        return all_chunks