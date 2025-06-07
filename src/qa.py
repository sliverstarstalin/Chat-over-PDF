from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List

import faiss
import numpy as np
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from embeddings import get_embed_model
from ingest import Chunk, extract_text

INDEX_DIR = Path("data/index")
RAW_DIR = Path("data/raw")

_index = None
_meta: List[dict] | None = None
_page_cache: dict[str, list[str]] = {}

# Conversation memory for chat sessions
memory = ConversationBufferMemory()


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
    "{history}\n"
    "CONTEXT:\n{context}\nQ:{question}\nA:"
)


@dataclass
class Citation:
    page: int
    snippet: str


@dataclass
class Answer:
    text: str
    citations: List[Citation]


def answer_question(query: str, *, local: bool = False) -> Answer:
    """Run RetrievalQA with conversation memory and optional local LLM."""
    chunks = retrieve(query)
    docs = [Document(page_content=c.text, metadata={"page": c.page}) for c in chunks]

    history = memory.load_memory_variables({}).get("history", "")
    prompt = PromptTemplate(
        input_variables=["context", "question", "history"],
        template=QA_PROMPT,
    )

    if local:
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="phi-3.5-mini",
        )
    else:
        llm = ChatOpenAI()

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    result = chain.invoke({"input_documents": docs, "question": query, "history": history})
    text = result["output_text"] if isinstance(result, dict) else str(result)

    pages = set(int(p) for p in re.findall(r"\(p (\d+)\)", text))
    citation_map = {}
    for c in chunks:
        if c.page in pages and c.page not in citation_map and c.text.strip():
            snippet = c.text.replace("\n", " ")[:200]
            citation_map[c.page] = snippet
    citations = [Citation(page=p, snippet=s) for p, s in citation_map.items()]

    memory.save_context({"question": query}, {"answer": text})
    return Answer(text=text, citations=citations)
