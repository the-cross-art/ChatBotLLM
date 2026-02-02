import os
import re
import uuid
import logging
from typing import Iterable, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_ollama import OllamaEmbeddings
from langgraph.store.postgres import PostgresStore

try:
    from pypdf import PdfReader  # optional, for PDFs
except Exception:
    PdfReader = None

DB_URI_RAG = "postgresql://imrannazir@localhost:5432/rag_knowledge"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag.ingest")


@dataclass
class DocMeta:
    doc_id: str
    source: str
    title: str | None = None


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed; cannot ingest PDF")
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n\n".join(texts)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_paragraphs(
    paragraphs: List[str], max_chars: int = 1200, overlap: int = 100
) -> List[str]:
    chunks: List[str] = []
    buf = []
    cur_len = 0
    for p in paragraphs:
        if cur_len + len(p) + 1 <= max_chars:
            buf.append(p)
            cur_len += len(p) + 1
        else:
            if buf:
                chunk = "\n\n".join(buf)
                chunks.append(chunk)
                # create overlap from end of previous buffer
                tail = chunk[-overlap:] if overlap > 0 else ""
                buf = [tail, p] if tail else [p]
                cur_len = len("\n\n".join(buf))
            else:
                # very long single paragraph; hard-split
                chunks.append(p[:max_chars])
                tail = p[:max_chars][-overlap:] if overlap > 0 else ""
                rest = p[max_chars:]
                buf = [tail, rest] if tail and rest else ([rest] if rest else [])
                cur_len = len("\n\n".join(buf))
    if buf:
        chunks.append("\n\n".join(buf))
    # final cleanup
    return [c.strip() for c in chunks if c.strip()]


def _extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        return _normalize_whitespace(_read_text_file(path))
    if ext in {".pdf"}:
        return _normalize_whitespace(_read_pdf_file(path))
    # fallback: try reading as text
    return _normalize_whitespace(_read_text_file(path))


def make_chunks(
    text: str, doc_id: str, source: str, title: str | None = None
) -> Iterable[Tuple[str, dict]]:
    paragraphs = _paragraphs(text)
    chunk_texts = _chunk_paragraphs(paragraphs)
    for i, chunk in enumerate(chunk_texts):
        meta = {
            "doc_id": doc_id,
            "source": source,
            "title": title,
            "section": f"chunk-{i+1}",
        }
        yield chunk, meta


def ingest_chunks(chunks: Iterable[Tuple[str, dict]]):
    """Persist precomputed chunks+metadata with embeddings into the RAG store."""
    with PostgresStore.from_conn_string(DB_URI_RAG) as store:
        store.setup()
        ns = ("rag", "global")

        count = 0
        for text, meta in chunks:
            emb = embeddings.embed_query(text)
            store.put(
                ns,
                str(uuid.uuid4()),
                {
                    "chunk": text,
                    "embedding": emb,
                    "doc_id": meta["doc_id"],
                    "source": meta["source"],
                    "title": meta.get("title"),
                    "section": meta.get("section"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            count += 1
        log.info("Ingested %d chunks", count)


def ingest_files(paths: List[str]):
    """High-level ingestion: read files, chunk, embed, and persist."""
    for path in paths:
        if not os.path.exists(path):
            log.warning("Skipping missing file: %s", path)
            continue
        title = os.path.basename(path)
        doc_id = str(uuid.uuid4())
        try:
            text = _extract_text(path)
            chunks = list(make_chunks(text, doc_id=doc_id, source=path, title=title))
            ingest_chunks(chunks)
            log.info("Finished ingest for %s (doc_id=%s)", path, doc_id)
        except Exception as e:
            log.error("Failed ingest for %s", path, exc_info=e)


if __name__ == "__main__":
    # Basic CLI: pass file paths to ingest
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python -m rag_ingestion.ingest <file1> [file2 ...]")
        _sys.exit(1)
    ingest_files(_sys.argv[1:])
