#!/usr/bin/env python3
"""
CLI script to fetch ArXiv metadata and optionally download PDFs.

Usage:
    python scripts/fetch_arxiv_data.py --max-results 100
    python scripts/fetch_arxiv_data.py --max-results 500 --download-pdfs
    python scripts/fetch_arxiv_data.py --batches 5 --results-per-batch 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.arxiv_fetcher import (
    batch_fetch_arxiv_metadata,
    save_metadatas,
    download_pdfs_from_metadatas_file,
)
from src.utils.logging_config import setup_logging

logger = setup_logging("fetch_arxiv_script", log_dir=Path("logs") / "scripts")


def main():
    """Main entry point for the ArXiv data fetching script."""
    parser = argparse.ArgumentParser(
        description="Fetch ArXiv metadata and optionally download PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 100 papers
  python scripts/fetch_arxiv_data.py --max-results 100

  # Fetch 500 papers and download PDFs
  python scripts/fetch_arxiv_data.py --max-results 500 --download-pdfs

  # Fetch in batches (5 batches of 100 each)
  python scripts/fetch_arxiv_data.py --batches 5 --results-per-batch 100

  # Fetch with custom delay between batches
  python scripts/fetch_arxiv_data.py --max-results 200 --delay 5.0
        """,
    )

    # Fetching options
    fetch_group = parser.add_argument_group("Fetching Options")
    fetch_group.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum number of papers to fetch (default: 100)",
    )
    fetch_group.add_argument(
        "--batches",
        type=int,
        default=None,
        help="Number of batches to fetch (overrides max-results calculation)",
    )
    fetch_group.add_argument(
        "--results-per-batch",
        type=int,
        default=100,
        help="Number of results per batch (default: 100)",
    )
    fetch_group.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds between batches (default: 5.0)",
    )

    # Action options
    action_group = parser.add_argument_group("Action Options")
    action_group.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download PDFs after fetching metadata",
    )
    action_group.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only fetch and save metadata (skip PDF downloads)",
    )

    args = parser.parse_args()

    try:
        # Step 1: Fetch metadata
        logger.info(
            "Starting ArXiv data fetch",
            extra={"script_args": vars(args)}
        )

        metadata = batch_fetch_arxiv_metadata(
            num_batches=args.batches,
            results_per_batch=args.results_per_batch,
            delay_seconds=args.delay,
            max_results=args.max_results if args.batches is None else None,
        )

        if not metadata:
            logger.warning("No metadata fetched")
            return 1

        logger.info(f"Fetched {len(metadata)} papers")

        # Step 2: Save metadata
        logger.info("Saving metadata to disk")
        save_metadatas(metadata)
        logger.info("Metadata saved successfully")

        # Step 3: Optionally download PDFs
        if args.download_pdfs and not args.metadata_only:
            logger.info("Starting PDF downloads")
            download_pdfs_from_metadatas_file()
            logger.info("PDF downloads completed")

        logger.info("ArXiv data fetch completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Fetch interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Failed to fetch ArXiv data", exc_info=exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

