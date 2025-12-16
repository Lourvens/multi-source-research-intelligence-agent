"""
Pydantic models for ArXiv-related data structures.
"""

from typing import List, Optional

from pydantic import BaseModel


class ArxivMetadata(BaseModel):
  """
  Metadata for a single ArXiv paper as produced by ArxivFetcher.
  """

  id: str
  title: str
  summary: Optional[str]
  authors: List[str]
  published: str
  updated: str
  links: List[str]
  categories: List[str]
  pdf_url: str
  fetched_at: str
  source: str


