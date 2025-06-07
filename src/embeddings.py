from __future__ import annotations

from typing import List

import numpy as np
from langchain_openai import OpenAIEmbeddings

from ingest import Chunk


class DummyEmbeddings:
    """Very small local embedding model for offline tests."""

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(16, dtype=float)
        for i, b in enumerate(text.encode("utf-8")):
            vec[i % 16] += b
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embed_model(name: str = "phi-3.5"):
    """Return a LangChain Embeddings instance switchable between local and OpenAI."""
    if name == "openai":
        return OpenAIEmbeddings()
    return DummyEmbeddings()


def embed_chunks(chunks: List[Chunk]) -> List[List[float]]:
    """Return dense vectors for every chunk.text using the model from 2-1."""
    embedder = get_embed_model()
    texts = [c.text for c in chunks]
    return embedder.embed_documents(texts)
