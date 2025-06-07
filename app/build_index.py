from __future__ import annotations

import argparse
from pathlib import Path
from glob import glob

from ingest import extract_text, chunk_pages
from embeddings import embed_chunks
from indexer import build_faiss_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs")
    parser.add_argument("pdfs", nargs="+", help="PDF files to index (supports glob patterns)")
    parser.add_argument("--out", default="data/index", help="Output directory")
    args = parser.parse_args()

    # Expand globs
    paths: list[Path] = []
    for p in args.pdfs:
        matches = glob(p)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(p))

    chunks = []
    for path in paths:
        pages = extract_text(Path(path))
        doc_chunks = chunk_pages(pages)
        for c in doc_chunks:
            c.source = Path(path).stem
        chunks.extend(doc_chunks)

    vectors = embed_chunks(chunks)
    build_faiss_index(chunks, vectors, Path(args.out))


if __name__ == "__main__":
    main()

