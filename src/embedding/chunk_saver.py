"""
Save processed chunks (with embeddings) to disk for reproducibility and caching.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from langchain_classic.schema import Document

from src import config
from src.utils.logging_config import setup_logging
from src.constants import DataSource, DATA_SOURCE_ARXIV

logger = setup_logging("chunk_saver", log_dir=Path("logs") / "embedding")


def save_processed_chunks(
    chunks: List[Document],
    source: str | DataSource = DATA_SOURCE_ARXIV,
    suffix: str = None
) -> Path:
    """
    Save processed chunks with embeddings to disk.
    
    Args:
        chunks: List of Document objects with embeddings in metadata
        source: Data source identifier (DataSource enum or string, e.g., DataSource.ARXIV)
        suffix: Optional suffix for filename (e.g., "abstracts", "full_text")
        
    Returns:
        Path to the saved file
    """
    # Convert enum to string value if needed
    source_str = source.value if isinstance(source, DataSource) else source
    if not chunks:
        logger.warning("No chunks to save")
        return None
    
    # Create output directory
    output_dir = config.PROCESSED_DATA_DIR / source_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{source_str}_chunks_{timestamp}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".json"
    
    file_path = output_dir / filename
    
    # Serialize chunks (convert embeddings from numpy arrays to lists)
    serialized_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": chunk.metadata.get("chunk_id", f"chunk_{i}"),
            "chunk_index": chunk.metadata.get("chunk_index", i),
            "page_content": chunk.page_content,
            "metadata": {}
        }
        
        # Copy metadata, handling embeddings specially
        for key, value in chunk.metadata.items():
            if key == "embedding":
                # Convert numpy array to list for JSON serialization
                if isinstance(value, np.ndarray):
                    chunk_data["metadata"][key] = value.tolist()
                elif isinstance(value, list):
                    chunk_data["metadata"][key] = value
                else:
                    logger.warning(f"Unexpected embedding type: {type(value)}")
            else:
                chunk_data["metadata"][key] = value
        
        serialized_chunks.append(chunk_data)
    
    # Save to file
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": source_str,
                    "timestamp": timestamp,
                    "total_chunks": len(chunks),
                    "chunks": serialized_chunks
                },
                f,
                ensure_ascii=False,
                indent=2
            )
        
        logger.info(
            "Saved processed chunks",
            extra={
                "file_path": str(file_path),
                "chunk_count": len(chunks),
                "source": source_str
            }
        )
        
        return file_path
        
    except Exception as exc:
        logger.error(
            "Failed to save processed chunks",
            extra={"file_path": str(file_path)},
            exc_info=exc
        )
        raise


def load_processed_chunks(file_path: Path) -> List[Document]:
    """
    Load processed chunks from disk.
    
    Args:
        file_path: Path to the JSON file containing chunks
        
    Returns:
        List of Document objects with embeddings restored
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        chunks = []
        for chunk_data in data.get("chunks", []):
            # Restore embedding to numpy array if present
            metadata = chunk_data.get("metadata", {}).copy()
            if "embedding" in metadata:
                metadata["embedding"] = np.array(metadata["embedding"])
            
            doc = Document(
                page_content=chunk_data["page_content"],
                metadata=metadata
            )
            chunks.append(doc)
        
        logger.info(
            "Loaded processed chunks",
            extra={
                "file_path": str(file_path),
                "chunk_count": len(chunks)
            }
        )
        
        return chunks
        
    except Exception as exc:
        logger.error(
            "Failed to load processed chunks",
            extra={"file_path": str(file_path)},
            exc_info=exc
        )
        raise

