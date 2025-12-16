# Scripts

Utility scripts for the Multi-Source Research Intelligence Platform.

## fetch_arxiv_data.py

Fetches ArXiv metadata and optionally downloads PDFs.

### Usage

```bash
# Basic usage: Fetch 100 papers
python scripts/fetch_arxiv_data.py --max-results 100

# Fetch 500 papers and download PDFs
python scripts/fetch_arxiv_data.py --max-results 500 --download-pdfs

# Fetch in batches (5 batches of 100 each = 500 papers)
python scripts/fetch_arxiv_data.py --batches 5 --results-per-batch 100

# Fetch with custom delay between batches (to respect rate limits)
python scripts/fetch_arxiv_data.py --max-results 200 --delay 10.0

# Only fetch metadata, skip PDF downloads
python scripts/fetch_arxiv_data.py --max-results 100 --metadata-only
```

### Options

- `--max-results N`: Maximum number of papers to fetch (default: 100)
- `--batches N`: Number of batches to fetch (overrides max-results calculation)
- `--results-per-batch N`: Number of results per batch (default: 100)
- `--delay SECONDS`: Delay in seconds between batches (default: 5.0)
- `--download-pdfs`: Download PDFs after fetching metadata
- `--metadata-only`: Only fetch and save metadata (skip PDF downloads)

### Examples

```bash
# Quick test: Fetch 10 papers
python scripts/fetch_arxiv_data.py --max-results 10

# Production: Fetch 1000 papers with PDFs
python scripts/fetch_arxiv_data.py --max-results 1000 --download-pdfs --delay 5.0
```

