from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path
import re
from typing import List

import faiss
import numpy as np
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from embeddings import get_embed_model
from ingest import Chunk, extract_text

INDEX_DIR = Path("data/index")
RAW_DIR = Path("data/raw")

_index = None
_meta: List[dict] | None = None
_page_cache: dict[str, list[str]] = {}


def _load_index() -> tuple[faiss.IndexFlatL2, List[dict]]:
    global _index, _meta
    if _index is None:
        index_path = INDEX_DIR / "index.faiss"
        meta_path = INDEX_DIR / "meta.json"
        _index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            _meta = json.load(f)
    assert _meta is not None
    return _index, _meta


def _load_pages(source: str) -> list[str]:
    if source not in _page_cache:
        pdf_path = RAW_DIR / f"{source}.pdf"
        _page_cache[source] = extract_text(pdf_path)
    return _page_cache[source]


def retrieve(query: str, k: int = 4) -> list[Chunk]:
    """Vector similarity search over the FAISS index."""
    index, meta = _load_index()
    embedder = get_embed_model()
    vec = np.array([embedder.embed_query(query)], dtype="float32")
    D, I = index.search(vec, k)
    results: list[Chunk] = []
    for idx in I[0]:
        if idx < 0:
            continue
        m = meta[idx]
        pages = _load_pages(m["source"])
        page_num = m["page"]
        text = pages[page_num - 1] if page_num - 1 < len(pages) else ""
        results.append(Chunk(text=text, page=page_num, source=m["source"]))
    return results


QA_PROMPT = (
    "Use only the CONTEXT to answer. Cite pages as (p ##).\n"
    "CONTEXT:\n{context}\nQ:{question}\nA:"
)

Answer = namedtuple("Answer", ["text", "citations"])


def answer_question(query: str) -> Answer:
    """Run RetrievalQA with prompt above; parse citations into a set of page ints."""
    chunks = retrieve(query)
    docs = [Document(page_content=c.text, metadata={"page": c.page}) for c in chunks]

    prompt = PromptTemplate(input_variables=["context", "question"], template=QA_PROMPT)
    llm = ChatOpenAI()
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    result = chain.invoke({"input_documents": docs, "question": query})
    text = result["output_text"] if isinstance(result, dict) else str(result)

    pages = set(int(p) for p in re.findall(r"\(p (\d+)\)", text))
    return Answer(text=text, citations=pages)
