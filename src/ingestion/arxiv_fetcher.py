import arxiv
import time
import math
import json
from datetime import datetime
from pathlib import Path

from src.utils.logging_config import setup_logging
from src.utils.download import download_pdf
from src import config
from src.types.arxiv import ArxivMetadata
from src.ingestion.constant import ARXIV_DEFAULT_QUERY, ARXIV_METADATA_SUBDIR, ARXIV_PDF_SUBDIR
from src.constants import DATA_SOURCE_ARXIV

logger = setup_logging("arxiv_fetcher", log_dir=Path("logs") / "ingestion")


def ArxivFetcher(max_results: int = 100) -> list:
  """
  Fetches the latest articles from the ArXiv.

  Args:
    max_results: The maximum number of articles to fetch.

  Returns:
    A list of articles with the following fields:
    - id: The ID of the article.
    - title: The title of the article.
    - summary: The summary of the article.
    - authors: The authors of the article.
    - published: The date the article was published.
    - updated: The date the article was updated.
    - links: The links to the article.
    - pdf_url: The URL of the PDF of the article.
  """
  logger.info("Fetching ArXiv articles", extra={"max_results": max_results})

  client = arxiv.Client()
  search = arxiv.Search(
    query=ARXIV_DEFAULT_QUERY,
    max_results=max_results,
    sort_by=arxiv.SortCriterion.Relevance,
  )

  metadata: list[dict] = []

  try:
    results = client.results(search)
    for result in results:
      metadata.append(
        {
          "id": result.get_short_id(),
          "title": result.title,
          "summary": result.summary if result.summary else None,
          "authors": [author.name for author in result.authors],
          "published": result.published.isoformat(),
          "updated": result.updated.isoformat(),
          "links": [link.href for link in result.links],
          "categories": [cat for cat in result.categories],
          "pdf_url": result.pdf_url,
          "fetched_at": datetime.now().isoformat(),
          "source": DATA_SOURCE_ARXIV,
        }
      )
  except Exception as exc:  # noqa: BLE001
    logger.error("Failed to fetch ArXiv articles", exc_info=exc)
    raise

  logger.info(
    "Fetched ArXiv articles successfully",
    extra={"count": len(metadata)},
  )
  return metadata


def batch_fetch_arxiv_metadata(
  num_batches: int | None = None,
  results_per_batch: int = 100,
  delay_seconds: float = 5.0,
  max_results: int | None = None,
):
  """
  Fetches the last articles from the ArXiv in batches of 100 using the ArxivFetcher
  function.
  
  Args:
    num_batches: How many batches (iterations) to fetch. If None, it is
      derived from max_results and results_per_batch.
    results_per_batch: How many results to fetch in each batch.
    delay_seconds: Optional delay between batches to respect rate limits.
    max_results: Optional total maximum number of articles to fetch
      (e.g. 2000). If provided and num_batches is None, then
      num_batches = ceil(max_results / results_per_batch).
  
  Returns:
    A list of lists, each containing the metadata for a batch of articles
    all metadatas in the batches
  """
  if num_batches is None:
    if max_results is None:
      num_batches = 1
    else:
      num_batches = math.ceil(max_results / results_per_batch)

  logger.info(
    "Starting batched ArXiv fetch",
    extra={
      "num_batches": num_batches,
      "results_per_batch": results_per_batch,
      "delay_seconds": delay_seconds,
      "max_results": max_results,
    },
  )

  metadata: list[list[dict]] = []

  for i in range(num_batches):
    if i > 0 and delay_seconds > 0:
      time.sleep(delay_seconds)

    batch = ArxivFetcher(max_results=results_per_batch)
    metadata.append(batch)

    logger.info(
      "Fetched ArXiv batch",
      
      extra={
        "batch_index": i,
        "batch_size": len(batch),
      },
    )

  flat_metadata = [item for batch in metadata for item in batch]
  logger.info(
    "Completed batched ArXiv fetch",
    extra={
      "total_items": len(flat_metadata),
      "batches": len(metadata),
    },
  )

  return flat_metadata

def save_metadatas(metadatas: list[dict]) -> None:
  """
  Save each metadata dictionary as a JSON file.

  Each metadata item is first validated against the ArxivMetadata Pydantic
  model, then written to:

      RAW_DATA_DIR / "arxiv_metadata" / "<id>.json"

  Args:
    metadatas: List of raw metadata dictionaries.
  """
  output_dir: Path = config.RAW_DATA_DIR / ARXIV_METADATA_SUBDIR
  output_dir.mkdir(parents=True, exist_ok=True)

  logger.info(
    "Saving ArXiv metadatas to disk",
    extra={"count": len(metadatas), "output_dir": str(output_dir)},
  )

  for meta in metadatas:
    try:
      arxiv_meta = ArxivMetadata(**meta)
      file_path = output_dir / f"{arxiv_meta.id}.json"
      with file_path.open("w", encoding="utf-8") as f:
        json.dump(arxiv_meta.model_dump(), f, ensure_ascii=False, indent=2)

      logger.info(
        "Saved ArXiv metadata",
        extra={"id": arxiv_meta.id, "path": str(file_path)},
      )
    except Exception as exc:  # noqa: BLE001
      logger.error(
        "Failed to save ArXiv metadata",
        extra={"raw_id": meta.get("id")},
        exc_info=exc,
      )

def download_pdfs_from_metadatas_file():
  """
  Download the PDFs from the metadatas file in the RAW_DATA_DIR / ARXIV_METADATA_SUBDIR directory.
  """
  input_dir: Path = config.RAW_DATA_DIR / ARXIV_METADATA_SUBDIR
  output_dir: Path = config.RAW_DATA_DIR / ARXIV_PDF_SUBDIR
  output_dir.mkdir(parents=True, exist_ok=True)

  for file in input_dir.glob("*.json"):
    with file.open("r", encoding="utf-8") as f:
      metadata = json.load(f)
      pdf_url = metadata.get("pdf_url")
      if pdf_url:
        pdf_path = output_dir / f"{file.stem}.pdf"
        if pdf_path.exists():
          logger.info(
            "PDF already downloaded, skipping",
            extra={
              "id": file.stem,
              "path": str(pdf_path),
            },
          )
          continue

        logger.info(
          "Downloading PDF from metadata",
          extra={
            "id": file.stem,
            "url": pdf_url,
            "destination": str(pdf_path),
          },
        )
        download_pdf(pdf_url, pdf_path)



# if __name__ == "__main__":
#   f"""
#   Get 50000 research paper about ARXIV_DEFAULT_QUERY, 
#   save the metadata to the RAW_DATA_DIR / ARXIV_METADATA_SUBDIR directory 
#   thendownload the PDFs to the RAW_DATA_DIR / ARXIV_PDF_SUBDIR directory.
#   """
#   results = batch_fetch_arxiv_metadata(num_batches=5, results_per_batch=1000)
#   save_metadatas(results)
#   download_pdfs_from_metadatas_file()