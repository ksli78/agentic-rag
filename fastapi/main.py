# -*- coding: utf-8 -*-
"""
ADAM FastAPI - Ingestion + Query

PDF -> text via PyPDF -> char-based chunking -> embeddings
"""

import io
import os
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

import lancedb
from lancedb import LanceDBConnection
from lancedb.pydantic import LanceModel, Vector

import ollama  # python client for Ollama

# ---------------------------
# Environment / Config
# ---------------------------
# Ollama (same as before)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2:latest")

# Vector DB
LANCEDB_URI = os.getenv("LANCEDB_URI", "/data/lancedb")

# Chunking knobs (characters)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

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


def extract_pdf_text_and_pages(pdf_bytes: bytes) -> Tuple[str, int]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts), len(reader.pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[int, int, str]]:
    start, n = 0, len(text)
    chunks: List[Tuple[int, int, str]] = []
    while start < n:
        end = min(n, start + size)
        chunks.append((1, 1, text[start:end]))
        if end == n:
            break
        start = max(0, end - overlap)
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


class DeleteDocumentPayload(BaseModel):
    doc_id: str
    confirm: str


class EraseDatabasePayload(BaseModel):
    confirm: str


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
    ch = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
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


@app.get("/documents")
def list_documents():
    conn = get_db()
    docs, _ = get_or_create_tables(conn)
    return {"documents": docs.to_list()}


@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)
    doc_rows = list(docs.search().where(f"doc_id == '{doc_id}'").to_list())
    if not doc_rows:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = list(chunks_tbl.search().where(f"doc_id == '{doc_id}'").to_list())
    return {"document": doc_rows[0], "chunks": chunks}


@app.post("/delete")
def delete_document(payload: DeleteDocumentPayload):
    if payload.confirm != "DELETE":
        raise HTTPException(status_code=400, detail="Confirmation mismatch")
    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)
    docs.delete(f"doc_id == '{payload.doc_id}'")
    chunks_tbl.delete(f"doc_id == '{payload.doc_id}'")
    return {"status": "deleted", "doc_id": payload.doc_id}


@app.post("/erase")
def erase_database(payload: EraseDatabasePayload):
    if payload.confirm != "ERASE":
        raise HTTPException(status_code=400, detail="Confirmation mismatch")
    conn = get_db()
    if "documents" in conn.table_names():
        conn.drop_table("documents")
    if "chunks" in conn.table_names():
        conn.drop_table("chunks")
    return {"status": "erased"}


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
    text, pcount = extract_pdf_text_and_pages(pdf_bytes)
    sha = sha256_bytes(pdf_bytes)
    # Stable id: prefer caller's doc_id, else hash of URL (or file hash)
    base_for_id = (source_url or sha)[:64]
    doc_id = doc_id or hashlib.sha1(base_for_id.encode("utf-8")).hexdigest()

    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)

    # unchanged short-circuit
    existing = list(docs.search().where(f"doc_id == '{doc_id}'").to_list())
    if existing and existing[0]["sha256"] == sha:
        return {"status": "skipped", "reason": "unchanged", "doc_id": doc_id}

    # delete previous chunks for this doc_id
    chunks_tbl.delete(f"doc_id == '{doc_id}'")

    # chunk + embed
    ch = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [c[2] for c in ch] if ch else [text]
    vecs = embed_texts(texts)

    rows = []
    # If chunking yielded nothing (empty doc), bail gracefully
    if not ch:
        ch = [(1, 1, text)]

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
    docs.add([
        {
            "doc_id": doc_id,
            "source_url": source_url,
            "title": title,
            "sha256": sha,
            "etag": etag,
            "last_modified": last_modified,
            "page_count": pcount,
            "ingested_at": datetime.utcnow().isoformat() + "Z",
        }
    ])

    return {"status": "ok", "doc_id": doc_id, "chunks": len(rows)}
