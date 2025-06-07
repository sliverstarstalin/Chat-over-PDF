from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from ingest import Chunk


def build_faiss_index(chunks: List[Chunk], vectors: List[List[float]], out_dir: Path) -> None:
    """Persist FAISS index plus metadata (page, source) under data/index/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.faiss"
    meta_path = out_dir / "meta.json"

    arr = np.array(vectors, dtype="float32")
    dim = arr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(arr)
    faiss.write_index(index, str(index_path))

    metadata = [{"page": c.page, "source": c.source} for c in chunks]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

