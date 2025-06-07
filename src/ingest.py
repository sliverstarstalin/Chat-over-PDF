from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import urllib.request

from pypdf import PdfReader

try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore
    convert_from_path = None


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
    """Return list of page-level strings using pypdf.PdfReader with OCR fallback."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            text = _ocr_page(path, i)
        pages.append(text)
    return pages


def _ocr_page(path: Path, page_number: int) -> str:
    """Return OCR text for a single page if pytesseract is available."""
    if pytesseract is None or convert_from_path is None:
        return ""
    try:  # pragma: no cover - heavy deps optional
        images = convert_from_path(str(path), first_page=page_number, last_page=page_number)
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception:
        return ""
    return ""


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
