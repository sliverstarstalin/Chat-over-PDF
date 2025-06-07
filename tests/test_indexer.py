from pathlib import Path
import sys
import json
import faiss
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ingest import extract_text, chunk_pages
from embeddings import embed_chunks, get_embed_model
from indexer import build_faiss_index


PDF_PATH = Path(__file__).resolve().parents[0] / "data" / "ashoka.pdf"


def test_build_and_search(tmp_path: Path):
    pages = extract_text(PDF_PATH)
    chunks = chunk_pages(pages)
    for c in chunks:
        c.source = PDF_PATH.stem

    vectors = embed_chunks(chunks)
    out_dir = tmp_path / "index"
    build_faiss_index(chunks, vectors, out_dir)

    index = faiss.read_index(str(out_dir / "index.faiss"))
    metadata = json.loads((out_dir / "meta.json").read_text())

    query_vec = np.array([get_embed_model().embed_query("Ashoka")], dtype="float32")
    D, I = index.search(query_vec, 1)
    assert I[0][0] >= 0
    assert len(metadata) > 0

