"""
Utility functions for downloading files (e.g., PDFs) from the web.
"""

from pathlib import Path
from typing import Final
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

from src.utils.logging_config import setup_logging


DOWNLOAD_CHUNK_SIZE: Final[int] = 8192

logger = setup_logging("download_utils")


def download_pdf(url: str, destination: Path) -> None:
  """
  Download a PDF from the given URL and save it to the destination path.

  This uses the Python standard library (urllib) to avoid extra
  dependencies. The destination directory is created if needed.

  Args:
    url: Direct URL to the PDF file.
    destination: Path where the PDF should be saved.
  """
  destination.parent.mkdir(parents=True, exist_ok=True)

  logger.info(
    f"Downloading PDF url:{url}",
    extra={"url": url, "destination": str(destination)},
  )

  try:
    with urlopen(url) as response, destination.open("wb") as out_file:
      while True:
        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
        if not chunk:
          break
        out_file.write(chunk)

    logger.info(
      "PDF downloaded successfully",
      extra={"destination": str(destination)},
    )
  except (HTTPError, URLError) as exc:
    logger.error(
      "Failed to download PDF",
      extra={"url": url, "destination": str(destination)},
      exc_info=exc,
    )
    raise


