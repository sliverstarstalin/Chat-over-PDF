from pathlib import Path
import sys

# Ensure src is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ingest import download_ncert, extract_text, chunk_pages, Chunk


SAMPLE_PDF = "https://raw.githubusercontent.com/mozilla/pdf.js/master/examples/learning/helloworld.pdf"


def test_extract_and_chunk():
    dest_dir = Path("data/raw")
    pdf_path = download_ncert(SAMPLE_PDF, dest_dir)
    assert pdf_path.exists()

    pages = extract_text(pdf_path)
    assert len(pages) > 0

    chunks = chunk_pages(pages)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(1 <= c.page <= len(pages) for c in chunks)
