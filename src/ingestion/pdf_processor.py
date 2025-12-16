"""
PDF text extraction using Langchain document loaders
"""


from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_classic.schema import Document

from src.utils.logging_config import setup_logging
from src import config
from src.ingestion.constant import ARXIV_PDF_SUBDIR
from src.constants import PDFLoaderType, PDF_LOADER_PYMUPDF, PDF_LOADER_PYPDF

logger = setup_logging("pdf_processor", log_dir=Path("logs") / "ingestion")


class PDFProcessor:
    """
    Extract text from PDF files using various strategies.
    """
    
    def __init__(self, loader_type: str | PDFLoaderType = PDFLoaderType.PYMUPDF):
        """
        Args:
            loader_type: Which PDF loader to use (PDFLoaderType enum or string)
                - PDFLoaderType.PYPDF: PyPDFLoader (slower, more reliable)
                - PDFLoaderType.PYMUPDF: PyMuPDFLoader (faster, better formatting)
        """
        # Convert enum to string value if needed
        self.loader_type = loader_type.value if isinstance(loader_type, PDFLoaderType) else loader_type
        logger.info(f"Initialized PDFProcessor with loader: {loader_type}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of LangChain Document objects (one per page)
            Each document has:
                - page_content: The text content
                - metadata: {"source": pdf_path, "page": page_num}
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            if self.loader_type == PDF_LOADER_PYMUPDF:
                loader = PyMuPDFLoader(str(pdf_path))
            else:
                loader = PyPDFLoader(str(pdf_path))
            
            # Load returns list of Document objects (one per page)
            documents = loader.load()
            
            logger.info(
                "Successfully extracted PDF text",
                extra={
                    "pdf": pdf_path.name,
                    "num_pages": len(documents),
                    "total_chars": sum(len(doc.page_content) for doc in documents)
                }
            )
            
            return documents
            
        except Exception as exc:
            logger.error(
                f"Failed to extract text from PDF: {pdf_path}",
                exc_info=exc
            )
            return []
    
    def process_all_pdfs(self, batch_size: int = 10) -> Dict[str, List[Document]]:
        """
        Process all PDFs in the PDF directory.
        
        Args:
            batch_size: Number of PDFs to process before logging progress
            
        Returns:
            Dict mapping arxiv_id -> List[Document]
        """
        pdf_dir = config.RAW_DATA_DIR / ARXIV_PDF_SUBDIR
        
        if not pdf_dir.exists():
            logger.error(f"PDF directory does not exist: {pdf_dir}")
            return {}
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs to process")
        
        all_documents = {}
        
        for i, pdf_path in enumerate(pdf_files):
            arxiv_id = pdf_path.stem  # Filename without .pdf
            
            # Extract text
            documents = self.extract_text_from_pdf(pdf_path)
            
            if documents:
                all_documents[arxiv_id] = documents
            
            # Progress logging
            if (i + 1) % batch_size == 0:
                logger.info(f"Processed {i + 1}/{len(pdf_files)} PDFs")
        
        logger.info(
            "Completed PDF processing",
            extra={
                "total_pdfs": len(pdf_files),
                "successful": len(all_documents),
                "failed": len(pdf_files) - len(all_documents)
            }
        )
        
        return all_documents
