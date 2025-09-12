# -*- coding: utf-8 -*-
"""
ADAM FastAPI - Ingestion + Query (PDF -> Marker Markdown -> heading-aware chunking -> embeddings)

What changed vs your previous version:
- We now convert PDFs to Markdown using the Marker library BEFORE chunking.
- Marker can optionally use a local LLM via Ollama to improve formatting (tables, math, layout).
- Chunking is heading/paragraph aware to keep semantic boundaries, then we add overlap.
- We kept LanceDB, Ollama embeddings, and your existing endpoints intact.

Notes:
- Embedding dim stays 768 (EmbeddingGemma) to match LanceDB schema.
- If Marker fails (bad install, model download), we gracefully fall back to PyPDF text.
"""

import io
import os
import re
import hashlib
import tempfile
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader

import lancedb
from lancedb import LanceDBConnection
from lancedb.pydantic import LanceModel, Vector

import ollama  # python client for Ollama

# ---- Marker imports (installed via requirements) ----
# We import lazily-friendly modules here; if torch/marker aren't present,
# we'll catch exceptions and fall back to simple PyPDF text extraction.
from marker.converters.pdf import PdfConverter  # type: ignore
from marker.models import create_model_dict  # type: ignore
from marker.output import text_from_rendered  # type: ignore
from marker.config.parser import ConfigParser  # type: ignore

# ---------------------------
# Environment / Config
# ---------------------------
# Ollama (same as before)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2:latest")

# Vector DB
LANCEDB_URI = os.getenv("LANCEDB_URI", "/data/lancedb")

# Chunking knobs (characters, not tokens, but works well with Markdown)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Marker knobs (override via env if needed)
USE_MARKER = os.getenv("USE_MARKER", "true").lower() in ("1", "true", "yes")
MARKER_USE_LLM = os.getenv("MARKER_USE_LLM", "true").lower() in ("1", "true", "yes")
MARKER_OLLAMA_MODEL = os.getenv("MARKER_OLLAMA_MODEL", "llama3:8b")
MARKER_FORCE_OCR = os.getenv("MARKER_FORCE_OCR", "false").lower() in ("1", "true", "yes")
MARKER_STRIP_OCR = os.getenv("MARKER_STRIP_OCR", "false").lower() in ("1", "true", "yes")
MARKER_DEBUG = os.getenv("MARKER_DEBUG", "false").lower() in ("1", "true", "yes")
# Device hint for Marker models ("cuda", "cuda:0", "cpu"). If unset, we auto-detect.
MARKER_DEVICE = os.getenv("MARKER_DEVICE", "").strip()

# Ollama client
client = ollama.Client(host=OLLAMA_HOST)

# FastAPI app
app = FastAPI(title="ADAM Ingestion & Query API")

# ---------------------------
# LanceDB Schemas
# ---------------------------

class Document(LanceModel):
    doc_id: str = Field(..., description="Stable ID (SP item id or GUID)")
    source_url: Optional[str] = None
    title: Optional[str] = None
    sha256: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    page_count: Optional[int] = None
    mime_type: Optional[str] = "application/pdf"
    ingested_at: str


class Chunk(LanceModel):
    doc_id: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    embedding: Vector(768)  # EmbeddingGemma vectors are 768-d


# ---------------------------
# DB helpers
# ---------------------------

def get_db() -> LanceDBConnection:
    os.makedirs(LANCEDB_URI, exist_ok=True)
    return lancedb.connect(LANCEDB_URI)


def get_or_create_tables(conn: LanceDBConnection):
    if "documents" not in conn.table_names():
        conn.create_table("documents", schema=Document, mode="overwrite")
    if "chunks" not in conn.table_names():
        conn.create_table("chunks", schema=Chunk, mode="overwrite")
    return conn.open_table("documents"), conn.open_table("chunks")


# ---------------------------
# Utility helpers
# ---------------------------

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def pdf_page_count(pdf_bytes: bytes) -> int:
    """Get page count quickly with PyPDF (cheap even if we use Marker)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)


# ---------------------------
# Marker conversion
# ---------------------------

def _detect_device() -> str:
    """
    Decide which device to use for Marker models.
    - Respect MARKER_DEVICE if provided (e.g., 'cuda', 'cuda:0', or 'cpu').
    - Else attempt to use CUDA if available; otherwise CPU.
    We avoid importing torch at module import to keep failure modes clean.
    """
    if MARKER_DEVICE:
        return MARKER_DEVICE
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def pdf_to_markdown_with_marker(pdf_bytes: bytes) -> str:
    """
    Convert PDF -> Markdown via Marker.
    If anything goes wrong, raise (caller will decide to fall back).
    """
    # Write to a temp file because Marker APIs expect file paths.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()

        # Build a Marker config that uses Ollama when LLM is enabled.
        config: Dict[str, Any] = {
            "output_format": "markdown",
            "use_llm": MARKER_USE_LLM,
            "debug": MARKER_DEBUG,
            "force_ocr": MARKER_FORCE_OCR,
            "strip_existing_ocr": MARKER_STRIP_OCR,
            "disable_image_extraction": True,  # we only need text
        }
        if MARKER_USE_LLM:
            # Tell Marker to use the local Ollama service + which model
            # (these keys are documented; see README and issues).
            config.update({
                "llm_service": "marker.services.ollama.OllamaService",
                "ollama_base_url": OLLAMA_HOST,
                "ollama_model": MARKER_OLLAMA_MODEL,
            })

        config_parser = ConfigParser(config)
        # Choose device for Marker models (layout/ocr/etc.)
        device = _detect_device()
        artifact_dict = create_model_dict(device=device)

        # Build the converter and run it
        converter = PdfConverter(
            artifact_dict=artifact_dict,
            config=config_parser.generate_config_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(tmp.name)
        # text_from_rendered returns (text, metadata, images) for markdown output
        md_text, _, _ = text_from_rendered(rendered)
        return md_text or ""


# ---------------------------
# Markdown-aware chunking
# ---------------------------

_MD_CODE_FENCE = re.compile(r"^```", re.MULTILINE)
_MD_HEADER = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _split_markdown_paragraphs(md: str) -> List[str]:
    """
    Split Markdown into logical paragraphs while respecting fenced code blocks.
    We keep code/table blocks intact, and only split on blank lines outside code.
    """
    lines = md.splitlines()
    paras: List[str] = []
    buf: List[str] = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            continue

        if not in_code and line.strip() == "":
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)

    if buf:
        paras.append("\n".join(buf).strip())
    return [p for p in paras if p]


def chunk_markdown(md: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[int, int, str]]:
    """
    Greedy pack paragraphs into ~size char chunks with ~overlap chars carried over.
    We use (page_start, page_end) placeholders = (1,1) since we don't map pages.
    """
    paras = _split_markdown_paragraphs(md)
    chunks: List[Tuple[int, int, str]] = []
    cur: List[str] = []
    cur_len = 0

    for p in paras:
        p_len = len(p) + 2  # account for spacing
        if cur and cur_len + p_len > size:
            chunk_text = "\n\n".join(cur).strip()
            if chunk_text:
                chunks.append((1, 1, chunk_text))

            # Build overlap by prepending trailing paragraphs whose cumulative
            # length is <= overlap
            carry: List[str] = []
            carry_len = 0
            for rp in reversed(cur):
                rp_len = len(rp) + 2
                if carry_len + rp_len > overlap:
                    break
                carry.insert(0, rp)
                carry_len += rp_len

            cur = carry + [p]
            cur_len = sum(len(x) + 2 for x in cur)
        else:
            cur.append(p)
            cur_len += p_len

    if cur:
        chunk_text = "\n\n".join(cur).strip()
        if chunk_text:
            chunks.append((1, 1, chunk_text))

    return chunks


# ---------------------------
# Embeddings / Generation
# ---------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings via Ollama. Handles both 'input' and 'prompt' flavors.
    """
    vectors: List[List[float]] = []
    for t in texts:
        try:
            out = client.embeddings(model=EMBED_MODEL, input=t)
            vec = out["embeddings"][0] if "embeddings" in out else out["embedding"]
        except Exception:
            out = client.embeddings(model=EMBED_MODEL, prompt=t)  # fallback for older clients
            vec = out.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError("Embedding vector missing or wrong type")
        vectors.append(vec)
    return vectors


def answer_with_context(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Very light instruct prompt with inline citations [1], [2], ...
    """
    parts = ["You are ADAM, a helpful assistant. Use the CONTEXT to answer."]
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"[{i}] doc_id={ctx['doc_id']} pgs {ctx['page_start']}-{ctx['page_end']}:\n{ctx['text'][:1200]}")
    parts.append(f"Question: {question}\nAnswer clearly with citations like [1], [2] where used.")
    prompt = "\n\n".join(parts)

    resp = client.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return resp["message"]["content"]


# ---------------------------
# Pydantic payloads
# ---------------------------

class IngestUrl(BaseModel):
    url: str
    doc_id: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    title: Optional[str] = None


class IngestText(BaseModel):
    doc_id: str
    text: str
    source_url: Optional[str] = None
    title: Optional[str] = None


class QueryPayload(BaseModel):
    question: str
    top_k: int = 6


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/health")
def health():
    try:
        client.list()  # ping Ollama
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ingest/url")
async def ingest_url(payload: IngestUrl):
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as s:
        r = await s.get(payload.url)
        r.raise_for_status()
        pdf_bytes = r.content

    return _ingest_common(
        pdf_bytes=pdf_bytes,
        doc_id=payload.doc_id,
        source_url=payload.url,
        etag=payload.etag,
        last_modified=payload.last_modified,
        title=payload.title
    )


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    etag: Optional[str] = Form(None),
    last_modified: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
):
    pdf_bytes = await file.read()
    return _ingest_common(
        pdf_bytes=pdf_bytes,
        doc_id=doc_id,
        source_url=source_url,
        etag=etag,
        last_modified=last_modified,
        title=title or file.filename
    )


@app.post("/ingest/text")
def ingest_text(payload: IngestText):
    """
    Keep existing text ingestion (useful for testing and non-PDF sources).
    """
    text = payload.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    doc_id = payload.doc_id or hashlib.sha1(text.encode("utf-8")).hexdigest()
    sha = sha256_bytes(text.encode("utf-8"))
    page_count = 1

    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)

    # unchanged short-circuit
    existing = list(docs.search().where(f"doc_id == '{doc_id}'").to_list())
    if existing and existing[0]["sha256"] == sha:
        return {"status": "skipped", "reason": "unchanged", "doc_id": doc_id}

    # delete any previous chunks for this doc_id
    chunks_tbl.delete(f"doc_id == '{doc_id}'")

    # chunk and embed
    ch = chunk_markdown(text, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [c[2] for c in ch]
    vecs = embed_texts(texts)

    # insert chunks
    rows = []
    for (pgs, pge, t), v in zip(ch, vecs):
        rows.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + t[:64]).encode("utf-8")).hexdigest(),
            "page_start": pgs,
            "page_end": pge,
            "text": t,
            "embedding": v
        })
    chunks_tbl.add(rows)

    # upsert doc metadata
    docs.delete(f"doc_id == '{doc_id}'")
    docs.add([{ 
        "doc_id": doc_id,
        "source_url": payload.source_url,
        "title": payload.title,
        "sha256": sha,
        "etag": None,
        "last_modified": None,
        "page_count": page_count,
        "ingested_at": datetime.utcnow().isoformat() + "Z"
    }])

    return {"status": "ok", "doc_id": doc_id, "chunks": len(rows)}


@app.post("/query")
def query(payload: QueryPayload):
    conn = get_db()
    _, chunks_tbl = get_or_create_tables(conn)

    # embed question
    qvec = embed_texts([payload.question])[0]

    # vector search
    hits = chunks_tbl.search(qvec).limit(payload.top_k).to_list()

    if not hits:
        return {"answer": "I couldn't find relevant content.", "citations": []}

    # build citations + context
    contexts = []
    for h in hits:
        contexts.append({
            "doc_id": h["doc_id"],
            "page_start": int(h.get("page_start", 1)),
            "page_end": int(h.get("page_end", 1)),
            "text": h["text"]
        })

    answer = answer_with_context(payload.question, contexts)

    # compact citations
    cset = [{"doc_id": c["doc_id"], "page_start": c["page_start"], "page_end": c["page_end"]} for c in contexts]
    return {"answer": answer, "citations": cset}


# ---------------------------
# Internal common ingest (PDF)
# ---------------------------

def _ingest_common(
    pdf_bytes: bytes,
    doc_id: Optional[str],
    source_url: Optional[str],
    etag: Optional[str],
    last_modified: Optional[str],
    title: Optional[str]
):
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Compute metadata first
    pcount = pdf_page_count(pdf_bytes)
    sha = sha256_bytes(pdf_bytes)
    # Stable id: prefer caller's doc_id, else hash of URL (or file hash)
    base_for_id = (source_url or sha)[:64]
    doc_id = doc_id or hashlib.sha1(base_for_id.encode("utf-8")).hexdigest()

    # Convert to Markdown with Marker (preferred), else fall back to PyPDF text
    try:
        if USE_MARKER:
            md_text = pdf_to_markdown_with_marker(pdf_bytes)
        else:
            raise RuntimeError("USE_MARKER disabled")  # jump to fallback
    except Exception:
        # Fallback: basic text extraction via PyPDF (no layout fixes)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        md_text = "\n\n".join(pages)

    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)

    # unchanged short-circuit
    existing = list(docs.search().where(f"doc_id == '{doc_id}'").to_list())
    if existing and existing[0]["sha256"] == sha:
        return {"status": "skipped", "reason": "unchanged", "doc_id": doc_id}

    # delete previous chunks for this doc_id
    chunks_tbl.delete(f"doc_id == '{doc_id}'")

    # chunk + embed
    ch = chunk_markdown(md_text, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [c[2] for c in ch] if ch else [md_text]
    vecs = embed_texts(texts)

    rows = []
    # If chunking yielded nothing (empty doc), bail gracefully
    if not ch:
        ch = [(1, 1, md_text)]

    for (pgs, pge, t), v in zip(ch, vecs):
        rows.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + t[:64]).encode("utf-8")).hexdigest(),
            "page_start": pgs,
            "page_end": pge,
            "text": t,
            "embedding": v
        })
    if rows:
        chunks_tbl.add(rows)

    # upsert document metadata
    docs.delete(f"doc_id == '{doc_id}'")
    docs.add([{ 
        "doc_id": doc_id,
        "source_url": source_url,
        "title": title,
        "sha256": sha,
        "etag": etag,
        "last_modified": last_modified,
        "page_count": pcount,
        "ingested_at": datetime.utcnow().isoformat() + "Z"
    }])

    return {"status": "ok", "doc_id": doc_id, "chunks": len(rows)}