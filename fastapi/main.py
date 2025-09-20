"""
Unified ADAM FastAPI application with deterministic retrieval and generation.

This implementation merges retrieval results from both LlamaIndex and LanceDB
according to a unified scoring pipeline, and exposes configuration knobs via
environment variables.  Setting deterministic generation parameters (seed,
temperature, top_p, repeat_penalty) ensures reproducible answers across runs.

The module retains ingestion endpoints for JSON, URL, upload and text inputs,
and uses LanceDB to store document and chunk metadata.  It also supports
classifying documents and extracting keywords via the LLM.

Usage:
    Set the appropriate environment variables (see the NEW CONFIG section)
    and run with `uvicorn main:app --reload`.
"""

import io
import os
import re
import json
import math
import hashlib
import tempfile
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

import lancedb
from lancedb import LanceDBConnection
from lancedb.pydantic import LanceModel, Vector

import ollama  # python client for Ollama

from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core import (
    ServiceContext,
    Document as LlamaDocument,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.langchain import LangchainEmbedding

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
# Basic Ollama and embedding models
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2:latest")
# Retrieval oversampling factor (per‑query) when selecting candidates
OVER_SAMPLE = int(os.getenv("OVER_SAMPLE", "10"))
# Maximum number of characters from a document to feed to the LLM
FULLDOC_MAX_CHARS = int(os.getenv("FULLDOC_MAX_CHARS", "12000"))
# Storage locations for LanceDB and LlamaIndex
LANCEDB_URI = os.getenv("LANCEDB_URI", "/data/lancedb")
LLAMA_INDEX_PATH = os.getenv("LLAMA_INDEX_PATH", "/data/llama_index")

# --------------------------- NEW CONFIG KNOBS ---------------------------
# Which retriever(s) to use: 'llama', 'lance', or 'hybrid' (both)
RETRIEVER = os.getenv("RETRIEVER", "llama").lower()
# Maximum number of chunks per document to include in answer context
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "3"))
# Scoring weights: lexical match, keyword match, category match
W_LEX = float(os.getenv("W_LEX", "1"))
W_KW  = float(os.getenv("W_KW", "1"))
W_CAT = float(os.getenv("W_CAT", "1"))
# Deterministic generation parameters
SEED = int(os.getenv("OLLAMA_SEED", os.getenv("SEED", "42")))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "1.0"))
GEN_REPEAT_PENALTY = float(os.getenv("GEN_REPEAT_PENALTY", "1.0"))
# Number of contexts to provide to the LLM (0 = no limit)
GEN_NUM_CTX = int(os.getenv("GEN_NUM_CTX", "6"))
# -----------------------------------------------------------------------

# Instantiate embedding model and service context
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
llama_embed_model = LangchainEmbedding(embedder)
service_context = ServiceContext.from_defaults(llm=None, embed_model=llama_embed_model)

# Chunking parameters (characters)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Noise filtering rules for PDF ingestion
NOISE_STARTS_WITH = [
    "This document contains",
    "Unauthorized use",
    "Uncontrolled if",
    "Before using this document",
    "Copyright",
    "All rights reserved",
    "CUI",
    "Controlled Unclassified",
    "Privacy Act",
    "Sensitive but unclassified",
]
NOISE_REGEX = [
    re.compile(r"\bproprietary information\b", re.IGNORECASE),
    re.compile(r"\bunauthorized use\b", re.IGNORECASE),
    re.compile(r"\buncontrolled if printed\b", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\bCUI\b", re.IGNORECASE),
    re.compile(r"\bControlled Unclassified\b", re.IGNORECASE),
    re.compile(r"\bPrivacy Act\b", re.IGNORECASE),
    re.compile(r"\bSensitive but unclassified\b", re.IGNORECASE),
]

# Initialize Ollama client and FastAPI app
client = ollama.Client(host=OLLAMA_HOST)
app = FastAPI(title="ADAM Ingestion & Query API")

# ---------------------------------------------------------------------------
# Database schemas and helpers
# ---------------------------------------------------------------------------

class Document(LanceModel):
    """Metadata for a single ingested document."""
    doc_id: str = Field(..., description="Stable ID (SP item id or GUID)")
    source_url: Optional[str] = None
    title: Optional[str] = None
    sha256: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    page_count: Optional[int] = None
    mime_type: Optional[str] = "application/pdf"
    ingested_at: str
    category: Optional[str] = None
    keywords: Optional[List[str]] = None


class Chunk(LanceModel):
    """A chunk of text from a document along with its embedding."""
    doc_id: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    embedding: Vector(768)


def get_db() -> LanceDBConnection:
    """Connect to the LanceDB database, creating directories as needed."""
    os.makedirs(LANCEDB_URI, exist_ok=True)
    return lancedb.connect(LANCEDB_URI)


def get_or_create_tables(conn: LanceDBConnection):
    """Ensure that the documents and chunks tables exist in LanceDB."""
    if "documents" not in conn.table_names():
        conn.create_table("documents", schema=Document, mode="overwrite")
    if "chunks" not in conn.table_names():
        conn.create_table("chunks", schema=Chunk, mode="overwrite")
    return conn.open_table("documents"), conn.open_table("chunks")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def remove_noise(text: str) -> str:
    """
    Remove lines that begin with known banner phrases or match regex patterns.
    Useful for cleaning PDF pages before chunking.
    """
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if any(stripped.lower().startswith(p.lower()) for p in NOISE_STARTS_WITH):
            continue
        if any(r.search(stripped) for r in NOISE_REGEX):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[int, int, str]]:
    """
    Split text into overlapping character chunks.  Returns a list of tuples
    (page_start, page_end, chunk_text).  For plain text ingestion we mark
    each chunk as page 1.
    """
    start, n = 0, len(text)
    chunks: List[Tuple[int, int, str]] = []
    while start < n:
        end = min(n, start + size)
        chunks.append((1, 1, text[start:end]))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def determine_question_category(question: str) -> Optional[str]:
    """
    Classify a question as asking about a policy, a procedure, or neither.
    If the text explicitly contains 'policy' or 'procedure' those are returned.
    Otherwise an LLM call with temperature=0 is used to decide.
    """
    lower_q = question.lower()
    if "policy" in lower_q:
        return "policy"
    if "procedure" in lower_q:
        return "procedure"
    system_prompt = (
        "You are an expert at classifying user questions about documents. "
        "Given a question, decide if the user is primarily interested in a policy, "
        "a procedure, or neither. Respond with exactly one word: policy, procedure, or other."
    )
    resp = client.chat(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        options={"temperature": 0.0},
    )
    ans = (resp["message"]["content"] or "").strip().lower()
    for w in ("policy", "procedure", "other"):
        if w in ans:
            return w
    return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Compute embeddings for a list of texts via Ollama's embedding API."""
    vectors: List[List[float]] = []
    for t in texts:
        try:
            out = client.embeddings(model=EMBED_MODEL, input=t)
            vec = out["embeddings"][0] if "embeddings" in out else out["embedding"]
        except Exception:
            out = client.embeddings(model=EMBED_MODEL, prompt=t)
            vec = out.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError("Embedding vector missing or wrong type")
        vectors.append(vec)
    return vectors


def load_llama_index() -> Optional[VectorStoreIndex]:
    """Load a persisted LlamaIndex from disk, or return None if unavailable."""
    if os.path.exists(LLAMA_INDEX_PATH):
        storage_context = StorageContext.from_defaults(persist_dir=LLAMA_INDEX_PATH)
        return load_index_from_storage(storage_context, service_context=service_context)
    return None


def _build_full_doc_context(doc_id: str, max_chars: int = FULLDOC_MAX_CHARS) -> Dict[str, Any]:
    """
    Assemble all chunks for a document into a single context dict, sorted by page
    and trimmed to max_chars.  Raises 404 if no chunks found.
    """
    conn = get_db()
    _, chunks_tbl = get_or_create_tables(conn)
    rows = list(chunks_tbl.search().where(f"doc_id == '{doc_id}'").to_list())
    if not rows:
        raise HTTPException(status_code=404, detail=f"No chunks found for doc_id={doc_id}")
    rows.sort(key=lambda r: (int(r.get("page_start", 1)), r.get("chunk_id", "")))
    full_text = "\n\n".join(r["text"] for r in rows if r.get("text"))
    if not full_text.strip():
        raise HTTPException(status_code=400, detail=f"Empty text for doc_id={doc_id}")
    max_chars = int(os.getenv("FULLDOC_MAX_CHARS", str(max_chars)))
    trimmed = full_text[:max_chars]
    first = rows[0]
    last = rows[-1]
    page_start = int(first.get("page_start", 1))
    page_end = int(last.get("page_end", page_start))
    return {
        "doc_id": doc_id,
        "page_start": page_start,
        "page_end": page_end,
        "text": trimmed,
    }


def extract_extra_metadata(markdown: str) -> Tuple[Optional[str], List[str]]:
    """
    Use the LLM to extract a category and a list of keywords from markdown content.
    Returns (category, keywords).  If the model fails to produce valid JSON,
    returns (None, []).
    """
    system_prompt = (
        "You are an expert document analyst. Given a document’s Markdown content, "
        "classify it as one of these categories: 'policy', 'procedure', or 'other'. "
        "Then extract 5–10 concise keywords (single words or short phrases) that best describe its subject. "
        "IMPORTANT: Return only a JSON object with the keys 'category' and 'keywords'—do not include explanations, markdown, or code fences."
    )
    excerpt = markdown[:5000]
    resp = client.chat(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": excerpt}],
        options={"temperature": 0.0},
    )
    content = resp["message"]["content"]
    json_match = re.search(r"```(?:json)?\s*({.*?})\s*```|({.*})", content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1) or json_match.group(2)
        json_str = json_str.replace("`", "").strip()
        try:
            data = json.loads(json_str)
            cat = data.get("category")
            kws = data.get("keywords", [])
            if not isinstance(kws, list):
                kws = []
            return cat, kws
        except json.JSONDecodeError:
            return None, []
    return None, []


# ---------------------------------------------------------------------------
# Answer generation & retrieval helpers
# ---------------------------------------------------------------------------

def answer_with_context(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Construct a prompt from multiple context snippets and call the language model
    to answer the question.  Only the first GEN_NUM_CTX contexts are used
    (unless GEN_NUM_CTX is 0).  Sampling parameters and a seed are taken from
    environment variables to make generation reproducible.  The assistant is
    instructed to use only the provided context and to cite sources inline.
    """
    limited = contexts[:GEN_NUM_CTX] if GEN_NUM_CTX else contexts
    parts: List[str] = [
        "You are ADAM, a helpful assistant. Use ONLY the CONTEXT provided to answer. "
        "If the answer is not present in the context, say you dont have enough information"
        " and direct user to check the SharePoint MS Docs located at https://portal.amentumspacemissions.com/MS/Pages/MSDefaultHomePage.aspx"

    ]
    for i, ctx in enumerate(limited, 1):
        snippet = ctx["text"][:FULLDOC_MAX_CHARS]
        parts.append(f"[{i}] doc_id={ctx['doc_id']} pgs {ctx['page_start']}-{ctx['page_end']}:\n{snippet}")
    parts.append(
        "Question: " + question + "\n"
        "Answer clearly and with detail, Add bracket citations like [1], [2] right after the sentences they support. your answer must be formatted in HTML with tables,"
        "lists, paragraphs"
    )
    prompt = "\n\n".join(parts)
    resp = client.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "repeat_penalty": GEN_REPEAT_PENALTY,
            "seed": SEED,
        },
    )
    return resp["message"]["content"]


def _compute_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors; return 0.0 on error."""
    try:
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 and n2:
            return dot / (n1 * n2)
    except Exception:
        pass
    return 0.0


def _retrieve_candidates(
    question: str,
    top_k: int,
    index: Optional[VectorStoreIndex],
    chunks_tbl,
    tokens: List[str],
    doc_map: Dict[str, Any],
    category: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Retrieve candidate chunks from LlamaIndex and/or LanceDB according to
    RETRIEVER, compute lexical/keyword/category heuristics, and return a
    sorted list of dictionaries containing text and metadata.  Sorting
    prioritises higher similarity scores, then weighted heuristic scores,
    followed by stable tie‑breakers (doc_id, node_id).
    """
    sample_k = max(1, top_k * OVER_SAMPLE)
    candidates: List[Dict[str, Any]] = []
    # LlamaIndex retrieval
    if RETRIEVER in ("llama", "hybrid") and index is not None:
        try:
            retriever = index.as_retriever(similarity_top_k=sample_k)
            results = retriever.retrieve(question)
        except Exception:
            results = []
        for node_with_score in results:
            node = getattr(node_with_score, "node", None) or node_with_score
            score = float(getattr(node_with_score, "score", 0.0) or 0.0)
            meta = getattr(node, "metadata", None) or {}
            text = getattr(node, "text", "") or ""
            doc_id = meta.get("doc_id") or ""
            page_start = meta.get("page_start", 1)
            page_end = meta.get("page_end", 1)
            node_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or ""
            candidates.append({
                "doc_id": doc_id,
                "text": text,
                "page_start": page_start,
                "page_end": page_end,
                "score": score,
                "node_id": node_id,
            })
    # LanceDB retrieval
    if RETRIEVER in ("lance", "hybrid") and chunks_tbl is not None:
        vecs = embed_texts([question])
        qvec = vecs[0] if vecs else None
        rows: List[Dict[str, Any]] = []
        if qvec is not None:
            try:
                rows = chunks_tbl.search(qvec).limit(sample_k).to_list()
            except Exception:
                try:
                    all_rows = chunks_tbl.search().to_list()
                except Exception:
                    all_rows = []
                scored = []
                for r in all_rows:
                    v = r.get("embedding")
                    if isinstance(v, list):
                        scored.append((_compute_similarity(qvec, v), r))
                scored.sort(key=lambda x: -x[0])
                rows = [r for _, r in scored[:sample_k]]
        for r in rows:
            v = r.get("embedding")
            candidates.append({
                "doc_id": r.get("doc_id", "") or "",
                "text": r.get("text", "") or "",
                "page_start": r.get("page_start", 1),
                "page_end": r.get("page_end", 1),
                "score": _compute_similarity(qvec, v) if qvec is not None and isinstance(v, list) else 0.0,
                "node_id": r.get("chunk_id", "") or "",
            })
    # Precompute docs whose keywords intersect the query tokens
    keyword_docs: set = set()
    for d_id, doc in doc_map.items():
        kws = [kw.lower() for kw in (doc.get("keywords") or [])]
        if kws and any(tok in kws for tok in tokens):
            keyword_docs.add(d_id)
    # Score and sort candidates
    scored: List[Dict[str, Any]] = []
    for cand in candidates:
        did = cand["doc_id"]
        text_lower = cand["text"].lower()
        lex_match = 1 if any(tok in text_lower for tok in tokens) else 0
        if did in keyword_docs:
            kw_score = 10
        else:
            kws = [kw.lower() for kw in (doc_map.get(did, {}).get("keywords") or [])]
            kw_score = sum(1 for kw in kws if kw in tokens)
        cat_score = 1 if category and doc_map.get(did, {}).get("category") == category else 0
        scored.append({**cand, "lex_match": lex_match, "kw_score": kw_score, "cat_score": cat_score})
    scored.sort(key=lambda c: (
        -c["score"],
        -(c["lex_match"] * W_LEX),
        -(c["kw_score"] * W_KW),
        -(c["cat_score"] * W_CAT),
        c["doc_id"],
        c["node_id"],
    ))
    return scored


# ---------------------------------------------------------------------------
# Pydantic payloads
# ---------------------------------------------------------------------------

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


class QueryFullPayload(BaseModel):
    question: str
    top_docs: int = 1


class QueryPayload(BaseModel):
    question: str
    top_k: int = 6
    doc_id: Optional[str] = None


class DeleteDocumentPayload(BaseModel):
    doc_id: str
    confirm: str


class EraseDatabasePayload(BaseModel):
    confirm: str


class IngestJsonPayload(BaseModel):
    file_name: str
    source_url: Optional[str] = None
    doc_id: str
    last_modified: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any]
    markdown: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Simple health check to verify the Ollama client is reachable."""
    try:
        client.list()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ingest/json")
def ingest_json(payload: IngestJsonPayload):
    """
    Accept a JSON object containing cleaned Markdown and metadata.  Extract
    keywords and category via the LLM, chunk and embed the text, and store
    everything in the database.  Returns document ID and metadata.
    """
    text = payload.markdown or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty markdown")
    base_for_id = payload.doc_id or hashlib.sha1((payload.file_name + (payload.source_url or "")).encode("utf-8")).hexdigest()
    doc_id = base_for_id
    sha = sha256_bytes(text.encode("utf-8"))
    page_count = 1
    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)
    existing = list(docs_tbl.search().where(f"doc_id == '{doc_id}'").to_list())
    if existing and existing[0]["sha256"] == sha:
        return {"status": "skipped", "reason": "unchanged", "doc_id": doc_id}
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [c[2] for c in chunks] if chunks else [text]
    embeddings = embed_texts(texts)
    rows: List[Dict[str, Any]] = []
    for (pgs, pge, chunk_text_val), vec in zip(chunks if chunks else [(1, 1, text)], embeddings):
        rows.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + chunk_text_val[:64]).encode("utf-8")).hexdigest(),
            "page_start": pgs,
            "page_end": pge,
            "text": chunk_text_val,
            "embedding": vec,
        })
    if rows:
        chunks_tbl.delete(f"doc_id == '{doc_id}'")
        chunks_tbl.add(rows)
    category, keywords = extract_extra_metadata(text)
    docs_tbl.delete(f"doc_id == '{doc_id}'")
    docs_tbl.add([
        {
            "doc_id": doc_id,
            "source_url": payload.source_url,
            "title": payload.title or payload.file_name,
            "sha256": sha,
            "last_modified": payload.last_modified,
            "page_count": page_count,
            "ingested_at": datetime.utcnow().isoformat() + "Z",
            "category": category,
            "keywords": keywords,
        }
    ])
    return {"status": "ok", "doc_id": doc_id, "chunks": len(rows), "category": category, "keywords": keywords}


@app.post("/ingest/url")
async def ingest_url(payload: IngestUrl):
    """Fetch a PDF from a URL, then ingest it via the common PDF ingest pipeline."""
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
        title=payload.title,
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
    """Ingest a PDF uploaded via multipart/form-data."""
    pdf_bytes = await file.read()
    return _ingest_common(
        pdf_bytes=pdf_bytes,
        doc_id=doc_id,
        source_url=source_url,
        etag=etag,
        last_modified=last_modified,
        title=title or file.filename,
    )


@app.post("/ingest/text")
def ingest_text(payload: IngestText):
    """Ingest plain text directly (for testing or non‑PDF sources)."""
    text = payload.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    doc_id = payload.doc_id or hashlib.sha1(text.encode("utf-8")).hexdigest()
    sha = sha256_bytes(text.encode("utf-8"))
    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)
    existing = list(docs_tbl.search().where(f"doc_id == '{doc_id}'").to_list())
    if existing and existing[0]["sha256"] == sha:
        return {"status": "skipped", "reason": "unchanged", "doc_id": doc_id}
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [c[2] for c in chunks]
    vecs = embed_texts(texts)
    rows: List[Dict[str, Any]] = []
    for (pgs, pge, t), v in zip(chunks, vecs):
        rows.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + t[:64]).encode("utf-8")).hexdigest(),
            "page_start": pgs,
            "page_end": pge,
            "text": t,
            "embedding": v,
        })
    chunks_tbl.delete(f"doc_id == '{doc_id}'")
    chunks_tbl.add(rows)
    docs_tbl.delete(f"doc_id == '{doc_id}'")
    docs_tbl.add([
        {
            "doc_id": doc_id,
            "source_url": payload.source_url,
            "title": payload.title,
            "sha256": sha,
            "etag": None,
            "last_modified": None,
            "page_count": 1,
            "ingested_at": datetime.utcnow().isoformat() + "Z",
        }
    ])
    return {"status": "ok", "doc_id": doc_id, "chunks": len(rows)}


@app.post("/query_full")
def query_full(payload: QueryFullPayload):
    """
    Unified retrieval for answering questions using entire documents.  Select the
    top N documents according to the unified scorer, assemble full contexts for
    each and return the LLM’s answer with citations.  If no relevant content
    exists, a fallback message is returned.
    """
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
    top_docs = payload.top_docs
    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)
    doc_map = {d["doc_id"]: d for d in docs_tbl.search().to_list()}
    index = load_llama_index() if RETRIEVER in ("llama", "hybrid") else None
    tokens = [tok for tok in re.findall(r"[a-zA-Z0-9']+", question.lower()) if tok]
    category = determine_question_category(question)
    scored = _retrieve_candidates(question, top_docs, index, chunks_tbl, tokens, doc_map, category)
    doc_ids: List[str] = []
    for cand in scored:
        did = cand["doc_id"]
        if did and did not in doc_ids:
            doc_ids.append(did)
        if len(doc_ids) >= top_docs:
            break
    if not doc_ids:
        return {"answer": "I couldn't find relevant content.", "citations": []}
    contexts: List[Dict[str, Any]] = []
    for did in doc_ids:
        try:
            ctx = _build_full_doc_context(did)
            contexts.append(ctx)
        except Exception:
            continue
    if not contexts:
        return {"answer": "I couldn't find relevant content.", "citations": []}
    answer = answer_with_context(question, contexts)
    citations: List[Dict[str, Any]] = []
    for ctx in contexts:
        meta = doc_map.get(ctx["doc_id"], {})
        citations.append({
            "doc_id": ctx["doc_id"],
            "title": meta.get("title"),
            "source_url": meta.get("source_url"),
            "page_start": ctx["page_start"],
            "page_end": ctx["page_end"],
        })
    return {"answer": answer, "citations": citations}


@app.post("/query")
def query(payload: QueryPayload):
    """
    Unified retrieval for answering questions using top‑scoring chunks from one or
    multiple documents.  If a doc_id is provided, the entire document is used
    as context.  Otherwise, candidates are gathered from the configured
    retriever(s), ranked, and up to MAX_CHUNKS_PER_DOC chunks per document are
    concatenated.  Returns the LLM’s answer and citations.
    """
    question = (payload.question or "").strip()
    top_k = payload.top_k
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)
    doc_map = {d["doc_id"]: d for d in docs_tbl.search().to_list()}
    # Full document lookup if doc_id is specified
    if payload.doc_id:
        did = payload.doc_id
        if did not in doc_map:
            raise HTTPException(status_code=404, detail=f"Document {did} not found")
        ctx = _build_full_doc_context(did)
        ans = answer_with_context(question, [ctx])
        return {
            "answer": ans,
            "citations": [
                {
                    "doc_id": did,
                    "title": doc_map[did].get("title"),
                    "source_url": doc_map[did].get("source_url"),
                    "page_start": ctx["page_start"],
                    "page_end": ctx["page_end"],
                }
            ],
        }
    index = load_llama_index() if RETRIEVER in ("llama", "hybrid") else None
    tokens = [tok for tok in re.findall(r"[a-zA-Z0-9']+", question.lower()) if tok]
    category = determine_question_category(question)
    scored = _retrieve_candidates(question, top_k, index, chunks_tbl, tokens, doc_map, category)
    if not scored:
        return {"answer": "I couldn't find relevant content.", "citations": []}
    doc_contexts: Dict[str, List[Dict[str, Any]]] = {}
    for cand in scored:
        did = cand["doc_id"]
        if not did:
            continue
        lst = doc_contexts.setdefault(did, [])
        if len(lst) < MAX_CHUNKS_PER_DOC:
            lst.append(cand)
        if len(doc_contexts) >= top_k and all(len(v) >= MAX_CHUNKS_PER_DOC for v in doc_contexts.values()):
            break
    contexts: List[Dict[str, Any]] = []
    for did in sorted(doc_contexts.keys()):
        chunks_for_doc = doc_contexts[did]
        combined_text = "\n\n".join(ch["text"] for ch in chunks_for_doc)
        first = chunks_for_doc[0]
        contexts.append({
            "doc_id": did,
            "page_start": first["page_start"],
            "page_end": first["page_end"],
            "text": combined_text,
        })
        if len(contexts) == top_k:
            break
    ans = answer_with_context(question, contexts)
    citations: List[Dict[str, Any]] = []
    for ctx in contexts:
        meta = doc_map.get(ctx["doc_id"], {})
        citations.append({
            "doc_id": ctx["doc_id"],
            "title": meta.get("title"),
            "source_url": meta.get("source_url"),
            "page_start": ctx["page_start"],
            "page_end": ctx["page_end"],
        })
    return {"answer": ans, "citations": citations}


@app.get("/documents")
def list_documents():
    """List all documents currently stored in LanceDB."""
    conn = get_db()
    docs, _ = get_or_create_tables(conn)
    return {"documents": list(docs.search().to_list())}


@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    """Fetch a single document and its associated chunks."""
    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)
    doc_rows = list(docs.search().where(f"doc_id == '{doc_id}'").to_list())
    if not doc_rows:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = list(chunks_tbl.search().where(f"doc_id == '{doc_id}'").to_list())
    return {"document": doc_rows[0], "chunks": chunks}


@app.post("/delete")
def delete_document(payload: DeleteDocumentPayload):
    """Delete a document and all its chunks, after confirmation."""
    if payload.confirm != "DELETE":
        raise HTTPException(status_code=400, detail="Confirmation mismatch")
    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)
    docs.delete(f"doc_id == '{payload.doc_id}'")
    chunks_tbl.delete(f"doc_id == '{payload.doc_id}'")
    return {"status": "deleted", "doc_id": payload.doc_id}


@app.post("/erase")
def erase_database(payload: EraseDatabasePayload):
    """Completely erase the database (both documents and chunks) after confirmation."""
    if payload.confirm != "ERASE":
        raise HTTPException(status_code=400, detail="Confirmation mismatch")
    conn = get_db()
    if "documents" in conn.table_names():
        conn.drop_table("documents")
    if "chunks" in conn.table_names():
        conn.drop_table("chunks")
    return {"status": "erased"}

def _ingest_common(
    *,
    pdf_bytes: bytes,
    doc_id: Optional[str],
    source_url: Optional[str],
    etag: Optional[str],
    last_modified: Optional[str],
    title: Optional[str],
) -> Dict[str, Any]:
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        loader = DoclingLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)
    if not documents:
        raise HTTPException(status_code=400, detail="Unable to parse PDF")
    for doc in documents:
        doc.page_content = remove_noise(doc.page_content)
    sha = sha256_bytes(pdf_bytes)
    base_for_id = (source_url or sha)[:64]
    doc_id = doc_id or hashlib.sha1(base_for_id.encode("utf-8")).hexdigest()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    split_docs = splitter.split_documents(documents)
    if not split_docs:
        raise HTTPException(status_code=400, detail="No text extracted")
    llama_docs: List[LlamaDocument] = []
    texts: List[str] = []
    for doc_obj in split_docs:
        meta = doc_obj.metadata or {}
        p_start = int(meta.get("page", 1))
        p_end = p_start
        text = doc_obj.page_content
        texts.append(text)
        llama_docs.append(
            LlamaDocument(
                text=text,
                metadata={
                    "doc_id": doc_id,
                    "page_start": p_start,
                    "page_end": p_end,
                },
            )
        )
    index = load_llama_index()
    if index is None:
        index = VectorStoreIndex.from_documents(llama_docs, service_context=service_context)
    else:
        for doc in llama_docs:
            index.insert(doc)
    index.storage_context.persist(persist_dir=LLAMA_INDEX_PATH)
    vecs = embedder.embed_documents(texts)
    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)
    chunks_tbl.delete(f"doc_id == '{doc_id}'")
    chunk_rows: List[Dict[str, Any]] = []
    for doc_obj, vec in zip(split_docs, vecs):
        meta = doc_obj.metadata or {}
        p_start = int(meta.get("page", 1))
        p_end = p_start
        content = doc_obj.page_content
        chunk_rows.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + content[:64]).encode("utf-8")).hexdigest(),
            "page_start": p_start,
            "page_end": p_end,
            "text": content,
            "embedding": vec,
        })
    if chunk_rows:
        chunks_tbl.add(chunk_rows)
    full_text = "\n\n".join([d.page_content for d in documents])
    category, keywords = extract_extra_metadata(full_text)
    page_count = len(documents)
    docs_tbl.delete(f"doc_id == '{doc_id}'")
    docs_tbl.add([
        {
            "doc_id": doc_id,
            "source_url": source_url,
            "title": title,
            "sha256": sha,
            "etag": etag,
            "last_modified": last_modified,
            "page_count": page_count,
            "ingested_at": datetime.utcnow().isoformat() + "Z",
            "category": category,
            "keywords": keywords,
        }
    ])
    return {"status": "ok", "doc_id": doc_id, "chunks": len(chunk_rows)}
