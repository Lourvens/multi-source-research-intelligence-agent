"""
Centralized constants and enums for the research agent system.

This module provides type-safe constants to replace string literals throughout
the codebase, improving maintainability and reducing errors.
"""

from enum import Enum

# Import ingestion constants for backward compatibility
from src.ingestion.constant import (
    ARXIV_DEFAULT_QUERY,
    ARXIV_METADATA_SUBDIR,
    ARXIV_PDF_SUBDIR,
)


class ChunkingStrategy(str, Enum):
    """Chunking strategies for document processing."""
    NONE = "none"
    RECURSIVE = "recursive"
    TOKEN = "token"


class ContentType(str, Enum):
    """Content types for documents."""
    ABSTRACT_ONLY = "abstract_only"
    FULL_PAPER = "full_paper"


class PDFLoaderType(str, Enum):
    """PDF loader types."""
    PYPDF = "pypdf"
    PYMUPDF = "pymupdf"


class DataSource(str, Enum):
    """Data source identifiers."""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    NEWS = "news"
    PATENTS = "patents"


# String constants for backward compatibility and convenience
CHUNK_STRATEGY_NONE = ChunkingStrategy.NONE.value
CHUNK_STRATEGY_RECURSIVE = ChunkingStrategy.RECURSIVE.value
CHUNK_STRATEGY_TOKEN = ChunkingStrategy.TOKEN.value

CONTENT_TYPE_ABSTRACT_ONLY = ContentType.ABSTRACT_ONLY.value
CONTENT_TYPE_FULL_PAPER = ContentType.FULL_PAPER.value

PDF_LOADER_PYPDF = PDFLoaderType.PYPDF.value
PDF_LOADER_PYMUPDF = PDFLoaderType.PYMUPDF.value

DATA_SOURCE_ARXIV = DataSource.ARXIV.value
DATA_SOURCE_PUBMED = DataSource.PUBMED.value
DATA_SOURCE_NEWS = DataSource.NEWS.value
DATA_SOURCE_PATENTS = DataSource.PATENTS.value

# Subdirectory names for data organization
PROCESSED_CHUNKS_SUBDIR = "chunks"
PROCESSED_DOCUMENTS_SUBDIR = "documents"

# Embedding model constants
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SENTENCE_TRANSFORMERS_PREFIX = "sentence-transformers/"

