"""
Load and combine metadata with PDF text content.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

from langchain_classic.schema import Document

from src.utils.logging_config import setup_logging
from src.ingestion.pdf_processor import PDFProcessor
from src import config
from src.ingestion.constant import ARXIV_METADATA_SUBDIR
from src.constants import (
    ContentType,
    CONTENT_TYPE_ABSTRACT_ONLY,
    CONTENT_TYPE_FULL_PAPER,
    DataSource,
    DATA_SOURCE_ARXIV,
    PDFLoaderType,
    PDF_LOADER_PYMUPDF,
    PROCESSED_DOCUMENTS_SUBDIR,
)

logger = setup_logging("document_loader", log_dir=Path("logs") / "ingestion")


class DocumentLoader:
    """
    Load documents with metadata and PDF text content combined.
    """
    
    def __init__(self):
        self.pdf_processor = PDFProcessor(loader_type=PDF_LOADER_PYMUPDF)
        self.metadata_dir = config.RAW_DATA_DIR / ARXIV_METADATA_SUBDIR
    
    def load_metadata(self, arxiv_id: str) -> Optional[Dict]:
        """
        Load metadata for a specific paper.
        
        Args:
            arxiv_id: ArXiv ID (e.g., "2301.12345")
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.metadata_dir / f"{arxiv_id}.json"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata not found for {arxiv_id}")
            return None
        
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Failed to load metadata for {arxiv_id}", exc_info=exc)
            return None
    
    def combine_pdf_with_metadata(
        self,
        arxiv_id: str,
        pdf_documents: List[Document],
        metadata: Dict
    ) -> Document:
        """
        Combine PDF text with metadata into single Document.
        
        Strategy:
        - Concatenate all pages into one text
        - Attach metadata to document
        - Create structured content: title + abstract + full_text
        
        Args:
            arxiv_id: ArXiv ID
            pdf_documents: List of Documents (one per page)
            metadata: Metadata dictionary
            
        Returns:
            Single Document with combined content and metadata
        """
        # Combine all pages into one text
        full_text = "\n\n".join(doc.page_content for doc in pdf_documents)
        
        # Structure the content
        # Format: Title -> Abstract -> Full Paper Text
        structured_content = f"""Title: {metadata['title']}

        Abstract:
        {metadata['summary']}

        Full Paper:
        {full_text}
      """
        
        # Create enriched metadata
        enriched_metadata = {
            "arxiv_id": arxiv_id,
            "title": metadata["title"],
            "authors": metadata["authors"],
            "published": metadata["published"],
            "updated": metadata["updated"],
            "categories": metadata.get("categories", []),
            "pdf_url": metadata["pdf_url"],
            "source": DATA_SOURCE_ARXIV,
            "content_type": CONTENT_TYPE_FULL_PAPER,
            "num_pages": len(pdf_documents),
            "total_chars": len(full_text)
        }
        
        # Create LangChain Document
        combined_doc = Document(
            page_content=structured_content,
            metadata=enriched_metadata
        )
        
        return combined_doc
    
    def load_all_documents(
        self,
        include_full_text: bool = True,
        max_documents: Optional[int] = None,
        save_to_disk: bool = False
    ) -> List[Document]:
        """
        Load all documents (metadata + PDF text).
        
        Args:
            include_full_text: If True, include PDF text. If False, abstract only.
            max_documents: Optional limit on number of documents to load
            save_to_disk: If True, save loaded documents to data/processed/arxiv/
            
        Returns:
            List of Document objects ready for chunking
        """
        logger.info(
            "Loading all documents",
            extra={
                "include_full_text": include_full_text,
                "max_documents": max_documents,
                "save_to_disk": save_to_disk
            }
        )
        
        # Get all metadata files
        metadata_files = list(self.metadata_dir.glob("*.json"))
        
        if max_documents:
            metadata_files = metadata_files[:max_documents]
        
        logger.info(f"Found {len(metadata_files)} metadata files")
        
        # Provide helpful error message if no metadata files found
        if len(metadata_files) == 0:
            error_msg = (
                f"No metadata files found in {self.metadata_dir}. "
                "Please fetch ArXiv metadata first by running:\n"
                "  1. from src.ingestion.arxiv_fetcher import batch_fetch_arxiv_metadata, save_metadatas\n"
                "  2. metadata = batch_fetch_arxiv_metadata(results_per_batch=100, num_batches=1)\n"
                "  3. save_metadatas(metadata['metadatas'])"
            )
            logger.error(error_msg)
            # Don't raise exception - return empty list to allow graceful handling
        
        documents = []
        
        # If including full text, process PDFs
        if include_full_text:
            pdf_docs_map = self.pdf_processor.process_all_pdfs()
        else:
            pdf_docs_map = {}
        
        # Load each document
        for i, metadata_file in enumerate(metadata_files):
            arxiv_id = metadata_file.stem
            
            # Load metadata
            metadata = self.load_metadata(arxiv_id)
            if not metadata:
                continue
            
            # Create document
            if include_full_text and arxiv_id in pdf_docs_map:
                # Full paper version
                doc = self.combine_pdf_with_metadata(
                    arxiv_id,
                    pdf_docs_map[arxiv_id],
                    metadata
                )
            else:
                # Abstract-only version (Phase 1 approach)
                doc = self._create_abstract_document(arxiv_id, metadata)
            
            documents.append(doc)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1}/{len(metadata_files)} documents")
        
        logger.info(
            "Completed document loading",
            extra={
                "total_documents": len(documents),
                "save_to_disk": save_to_disk
            }
        )
        
        # Save loaded documents to disk if requested (for testing and monitoring)
        if save_to_disk and documents:
            saved_paths = self._save_loaded_documents(documents, include_full_text)
            if saved_paths:
                logger.info(
                    "Saved loaded documents for testing/monitoring",
                    extra={
                        "saved_files": len(saved_paths),
                        "document_count": len(documents),
                        "output_dir": str(saved_paths[0].parent) if saved_paths else None,
                        "content_type": "full_paper" if include_full_text else "abstract_only"
                    }
                )
            else:
                logger.warning("Failed to save loaded documents (continuing anyway)")
        
        return documents
    
    def _create_abstract_document(self, arxiv_id: str, metadata: Dict) -> Document:
        """
        Create document from abstract only (no PDF text).
        """
        content = f"""Title: {metadata['title']}

        Abstract:
        {metadata['summary']}
        """
        
        doc_metadata = {
            "arxiv_id": arxiv_id,
            "title": metadata["title"],
            "authors": metadata["authors"],
            "published": metadata["published"],
            "updated": metadata["updated"],
            "categories": metadata.get("categories", []),
            "pdf_url": metadata["pdf_url"],
            "source": DATA_SOURCE_ARXIV,
            "content_type": CONTENT_TYPE_ABSTRACT_ONLY,
            "total_chars": len(content)
        }
        
        return Document(page_content=content, metadata=doc_metadata)
    
    def _save_loaded_documents(
        self,
        documents: List[Document],
        include_full_text: bool
    ) -> List[Path]:
        """
        Save loaded documents to disk, one file per document.
        Each file contains metadata at the top, then the document content.
        
        Args:
            documents: List of loaded Document objects
            include_full_text: Whether full text was included
            
        Returns:
            List of paths to saved files
        """
        if not documents:
            return []
        
        # Create output directory
        output_dir = config.PROCESSED_DATA_DIR / DATA_SOURCE_ARXIV / PROCESSED_DOCUMENTS_SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = "full_text" if include_full_text else "abstracts"
        saved_paths = []
        
        # Save each document as a separate file
        for i, doc in enumerate(documents):
            arxiv_id = doc.metadata.get("arxiv_id", f"unknown_{i}")
            
            # Create filename: arxiv_id_timestamp_suffix.json
            filename = f"{arxiv_id}_{timestamp}_{suffix}.json"
            file_path = output_dir / filename
            
            # Prepare document data with metadata at the top
            doc_data = {
                # Metadata section (at the top)
                "metadata": {
                    "arxiv_id": doc.metadata.get("arxiv_id"),
                    "title": doc.metadata.get("title"),
                    "authors": doc.metadata.get("authors", []),
                    "published": doc.metadata.get("published"),
                    "updated": doc.metadata.get("updated"),
                    "categories": doc.metadata.get("categories", []),
                    "pdf_url": doc.metadata.get("pdf_url"),
                    "source": doc.metadata.get("source", DATA_SOURCE_ARXIV),
                    "content_type": doc.metadata.get("content_type"),
                    "num_pages": doc.metadata.get("num_pages", 0),
                    "total_chars": doc.metadata.get("total_chars", len(doc.page_content)),
                    "saved_at": timestamp
                },
                # Document content
                "page_content": doc.page_content
            }
            
            # Save individual document file
            try:
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        doc_data,
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
                
                saved_paths.append(file_path)
                
            except Exception as exc:
                logger.error(
                    f"Failed to save document {arxiv_id}",
                    extra={"file_path": str(file_path), "arxiv_id": arxiv_id},
                    exc_info=exc
                )
        
        if saved_paths:
            logger.info(
                "Saved loaded documents (one file per document)",
                extra={
                    "saved_count": len(saved_paths),
                    "total_count": len(documents),
                    "output_dir": str(output_dir),
                    "content_type": "full_paper" if include_full_text else "abstract_only"
                }
            )
        
        return saved_paths