"""
Configuration management for the Research RAG project using Pydantic Settings.

Loads settings from environment variables and .env file with validation and type safety.
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',  # Ignore extra env vars not defined in model
    )
    
    # API Keys
    OPENAI_API_KEY: str | None = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: str | None = Field(default=None, description="Anthropic API key")
    
    # ArXiv Configuration
    ARXIV_MAX_RESULTS: int = Field(default=100, ge=1, description="Maximum number of ArXiv results to fetch")
    ARXIV_DELAY_SECONDS: float = Field(default=3.0, ge=0.0, description="Delay between ArXiv API calls (seconds)")
    
    # Document Processing
    CHUNK_SIZE: int = Field(default=1000, ge=1, description="Default chunk size for document splitting")
    CHUNK_OVERLAP: int = Field(default=200, ge=0, description="Overlap between chunks")
    
    # Vector Store
    VECTOR_DB_PATH: str | None = Field(default=None, description="Path to vector database directory")
    COLLECTION_NAME: str = Field(default="arxiv_papers", description="Default ChromaDB collection name")
    
    # Embedding Model
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    
    # LLM Configuration
    LLM_MODEL: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = Field(default=5, ge=1, description="Number of top documents to retrieve")
    
    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard levels."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        # Validate that at least one API key is provided
        if not self.OPENAI_API_KEY and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "Either OPENAI_API_KEY or ANTHROPIC_API_KEY must be set in .env file"
            )


# Initialize settings instance
_settings = Settings()

# Export settings as module-level constants for backward compatibility
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
VECTOR_DB_DIR = Path(_settings.VECTOR_DB_PATH) if _settings.VECTOR_DB_PATH else PROJECT_ROOT / "vector_db"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  CACHE_DIR, VECTOR_DB_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = _settings.OPENAI_API_KEY
ANTHROPIC_API_KEY = _settings.ANTHROPIC_API_KEY

# ArXiv Configuration
ARXIV_MAX_RESULTS = _settings.ARXIV_MAX_RESULTS
ARXIV_DELAY_SECONDS = _settings.ARXIV_DELAY_SECONDS

# Document Processing
CHUNK_SIZE = _settings.CHUNK_SIZE
CHUNK_OVERLAP = _settings.CHUNK_OVERLAP

# Vector Store
VECTOR_DB_PATH = str(VECTOR_DB_DIR)
COLLECTION_NAME = _settings.COLLECTION_NAME

# Embedding Model
EMBEDDING_MODEL = _settings.EMBEDDING_MODEL

# LLM Configuration
LLM_MODEL = _settings.LLM_MODEL
LLM_TEMPERATURE = _settings.LLM_TEMPERATURE

# Retrieval Configuration
RETRIEVAL_TOP_K = _settings.RETRIEVAL_TOP_K

# Logging
LOG_LEVEL = _settings.LOG_LEVEL

# Export settings instance for advanced usage
settings = _settings
