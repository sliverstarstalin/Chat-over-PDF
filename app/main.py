from __future__ import annotations

import csv
import time
import uuid
from pathlib import Path
import urllib.parse
import argparse

import streamlit as st

from ingest import extract_text, chunk_pages
from embeddings import embed_chunks
from indexer import build_faiss_index
from qa import answer_question

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--local", action="store_true", help="Use local phi-3.5 server")
ARGS, _ = parser.parse_known_args()


RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")
LOG_PATH = Path("data/logs.csv")


def build_index() -> None:
    """Build FAISS index from all PDFs under RAW_DIR."""
    pdf_paths = list(RAW_DIR.glob("*.pdf"))
    chunks = []
    for path in pdf_paths:
        pages = extract_text(path)
        doc_chunks = chunk_pages(pages)
        for c in doc_chunks:
            c.source = path.stem
        chunks.extend(doc_chunks)
    if not chunks:
        return
    vectors = embed_chunks(chunks)
    build_faiss_index(chunks, vectors, INDEX_DIR)


def log_interaction(user: str, question: str, latency: float) -> None:
    """Append {user, question, latency} to LOG_PATH."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "question", "latency"])
        if write_header:
            writer.writeheader()
        writer.writerow({"user": user, "question": question, "latency": f"{latency:.2f}"})


st.set_page_config(
    page_title="NCERT Study Buddy",
    layout="wide",
    theme={"base": "dark"},
)

if "session_id" not in st.session_state:
    user_id = st.user.get("id", str(uuid.uuid4()))
    st.session_state["session_id"] = user_id

left_col, mid_col, right_col = st.columns(3)

with left_col:
    st.header("Documents")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs = sorted(p.stem for p in RAW_DIR.glob("*.pdf"))
    selected = st.multiselect("Select PDF(s)", docs, default=docs)

    uploads = st.file_uploader("Upload new PDF(s)", type="pdf", accept_multiple_files=True)
    if uploads:
        with st.spinner("Indexing uploaded PDFs..."):
            for up in uploads:
                path = RAW_DIR / up.name
                with open(path, "wb") as f:
                    f.write(up.getbuffer())
            build_index()
        st.experimental_rerun()

with mid_col:
    st.header("Ask a question")
    question = st.text_input("Question")
    ask = st.button("Submit")

with right_col:
    st.header("Answer")
    if ask and question:
        start = time.perf_counter()
        ans = answer_question(question, local=ARGS.local)
        elapsed = time.perf_counter() - start

        st.markdown(ans.text)

        with st.expander("Sources"):
            for cit in ans.citations:
                pdf = selected[0] if selected else docs[0] if docs else "document"
                file_path = RAW_DIR / f"{pdf}.pdf"
                file_url = urllib.parse.quote(str(file_path))
                highlight = urllib.parse.quote(cit.snippet)
                viewer = f"viewer.html?file={file_url}&page={cit.page}&highlight={highlight}"
                st.markdown(f"- [Page {cit.page}]({viewer})")

        log_interaction(st.session_state["session_id"], question, elapsed)
