from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import urllib.request

from pypdf import PdfReader


def download_ncert(url: str, dest_dir: Path) -> Path:
    """Stream a PDF to data/raw/, skip if already exists, return file path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    path = dest_dir / filename
    if path.exists():
        return path
    with urllib.request.urlopen(url) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)
    return path


def extract_text(path: Path) -> list[str]:
    """Return list of page-level strings using pypdf.PdfReader."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


@dataclass
class Chunk:
    text: str
    page: int
    source: str  # file stem


def chunk_pages(pages: list[str], max_chars: int = 1000) -> list[Chunk]:
    """Split pages into overlapping chunks ≤ max_chars with ~200-char overlap."""
    overlap = 200
    chunks: list[Chunk] = []
    for page_num, text in enumerate(pages, start=1):
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk_text = text[start:end]
            chunks.append(Chunk(text=chunk_text, page=page_num, source="document"))
            if len(text) <= end:
                break
            start += max_chars - overlap
    return chunks
