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
import tempfile

import json
import re

# from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# ---------------------------
# Environment / Config
# ---------------------------
# Ollama (same as before)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL","sentence-transformers/all-mpnet-base-v2")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2:latest")

# Vector DB
LANCEDB_URI = os.getenv("LANCEDB_URI", "/data/lancedb")
# Path on the Docker volume where the FAISS index will be stored
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/data/faiss_index")

# Create an embeddings object once; OllamaEmbeddings wraps the same model you’re using
embedder = HuggingFaceEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)

# Chunking knobs (characters)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

#NOISE FILTERING 
NOISE_STARTS_WITH = [
    "This document contains",     # proprietary info header
    "Unauthorized use",           # part of the banner
    "Uncontrolled if",            # uncontrolled copy notice
    "Before using this document",
    "Copyright",
    "All rights reserved",
    "CUI",
    "Controlled Unclassified",
    "Privacy Act",
    "Sensitive but unclassified"
]

NOISE_REGEX = [
    re.compile(r"\bproprietary information\b", re.IGNORECASE),
    re.compile(r"\bunauthorized use\b", re.IGNORECASE),
    re.compile(r"\buncontrolled if printed\b", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\bCUI\b", re.IGNORECASE),
    re.compile(r"\bControlled Unclassified\b", re.IGNORECASE),
    re.compile(r"\bPrivacy Act\b", re.IGNORECASE),
    re.compile(r"\bSensitive but unclassified\b", re.IGNORECASE)
]


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
    # New fields
    category: Optional[str] = None  # e.g. “policy”, “procedure”
    keywords: Optional[List[str]] = None  # extracted key phrases


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
def remove_noise(text: str) -> str:
    """
    Remove lines that start with known noise phrases or match noise patterns.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # skip empty lines early to reduce pattern checks
        if not stripped:
            cleaned_lines.append(line)
            continue
        # check if the line starts with any banned phrase
        if any(stripped.lower().startswith(p.lower()) for p in NOISE_STARTS_WITH):
            continue
        # check regex patterns
        if any(r.search(stripped) for r in NOISE_REGEX):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

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

def determine_question_category(question: str) -> Optional[str]:
    """
    Quickly identify whether the question is about a policy or a procedure.
    If the text explicitly contains 'policy' or 'procedure', return that.
    Otherwise use the LLM to decide (returns 'policy', 'procedure', or 'other').
    """
    lower_q = question.lower()
    if "policy" in lower_q:
        return "policy"
    if "procedure" in lower_q:
        return "procedure"

    # Fall back to LLM classification
    system_prompt = (
        "You are an expert at classifying user questions about documents.  "
        "Given a question, decide if the user is primarily interested in a policy, "
        "a procedure, or neither.  Respond with exactly one word: policy, procedure, or other."
    )
    resp = client.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        options={"temperature": 0.0}
    )
    ans = (resp["message"]["content"] or "").strip().lower()
    for w in ("policy", "procedure", "other"):
        if w in ans:
            return w
    return None

def get_faiss_index() -> Optional[FAISS]:
    """
    Load the FAISS index from disk if it exists; return None if not.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        # embedder is now your HuggingFaceEmbeddings instance
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embedder,
            allow_dangerous_deserialization=True
        )
    return None
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

def extract_extra_metadata(markdown: str) -> Tuple[Optional[str], List[str]]:
    """
    Use the LLM to classify the document type (policy/procedure/other)
    and extract keywords.  The prompt instructs the model to return
    only a JSON object.  The parser is tolerant of extra text.
    """
    system_prompt = (
        "You are an expert document analyst.  Given a document’s Markdown content, "
        "classify it as one of these categories: 'policy', 'procedure', or 'other'.  "
        "Then extract 5–10 concise keywords (single words or short phrases) that best "
        "describe its subject.  IMPORTANT: Return only a JSON object with the keys "
        "'category' and 'keywords'—do not include explanations, markdown, or code fences."
    )

    # Send a truncated excerpt to keep token usage reasonable
    excerpt = markdown[:5000]
    resp = client.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": excerpt}
        ],
        options={"temperature": 0.0}  # low temperature improves determinism
    )
    content = resp["message"]["content"]

    print(content)

    json_match = re.search(r"```(?:json)?\s*({.*?})\s*```|({.*})", content, re.DOTALL)
    
    if json_match:
        # Extract the content of the matched group
        json_str = json_match.group(1) or json_match.group(2)
        
        # Clean up the string to ensure it's valid JSON
        json_str = json_str.replace("`", "").strip()

        try:
            data = json.loads(json_str)
            cat = data.get("category")
            kws = data.get("keywords", [])
            
            # Additional validation to ensure keywords is a list
            if not isinstance(kws, list):
                kws = []
                
            return cat, kws
            
        except json.JSONDecodeError as e:
            # Log the error for debugging
            print(f"Failed to decode JSON: {e}")
            print(f"Attempted to parse: {json_str}")
            return None, []
    
    # If no JSON pattern is found
    return None, []


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

class IngestJsonPayload(BaseModel):
    """
    Expected payload for /ingest/json.  This mirrors the JSON produced
    by the C# SharePoint crawler.
    """
    file_name: str
    source_url: Optional[str] = None
    doc_id: str
    last_modified: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any]
    markdown: str

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

@app.post("/ingest/json")
def ingest_json(payload: IngestJsonPayload):
    """
    Accept a JSON object containing cleaned Markdown and original metadata.
    Uses LLM to extract extra metadata (category and keywords) and stores
    both the chunks and the metadata in the database.
    """
    text = payload.markdown or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty markdown")

    # Derive a stable doc_id (prefer caller’s ID, else file name hash)
    base_for_id = payload.doc_id or hashlib.sha1((payload.file_name + (payload.source_url or "")).encode("utf-8")).hexdigest()
    doc_id = base_for_id

    # Compute SHA of the markdown for change detection
    sha = sha256_bytes(text.encode("utf-8"))
    page_count = 1  # unknown; treat as 1 since we no longer have page boundaries

    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)

    # If existing doc has same SHA, skip
    existing = list(docs.search().where(f"doc_id == '{doc_id}'").to_list())
    if existing and existing[0]["sha256"] == sha:
        return {"status": "skipped", "reason": "unchanged", "doc_id": doc_id}

    # Delete any existing chunks for this doc
    chunks_tbl.delete(f"doc_id == '{doc_id}'")

    # Chunk and embed
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [c[2] for c in chunks] if chunks else [text]
    embeddings = embed_texts(texts)

    rows = []
    for (pgs, pge, chunk_text_val), vec in zip(chunks if chunks else [(1, 1, text)], embeddings):
        rows.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + chunk_text_val[:64]).encode("utf-8")).hexdigest(),
            "page_start": pgs,
            "page_end": pge,
            "text": chunk_text_val,
            "embedding": vec
        })
    if rows:
        chunks_tbl.add(rows)

    # Extract extra metadata via LLM (category + keywords)
    category, keywords = extract_extra_metadata(text)

    # Upsert document metadata
    docs.delete(f"doc_id == '{doc_id}'")
    docs.add([{
        "doc_id": doc_id,
        "source_url": payload.source_url,
        "title": payload.title or payload.file_name,
        "sha256": sha,
        "last_modified": payload.last_modified,
        "page_count": page_count,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "category": category,
        "keywords": keywords
    }])

    return {"status": "ok", "doc_id": doc_id, "chunks": len(rows), "category": category, "keywords": keywords}


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
# -----------------------------------------
# Query Endpoint with category + keyword logic
# -----------------------------------------
@app.post("/query")
def query(payload: QueryPayload):
    """
    Query the FAISS index and return an answer with citations.
    Uses lexical match, keyword match, and category match to boost results.
    """
    question = payload.question or ""
    top_k = payload.top_k
    if not question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    # Load FAISS index
    index = get_faiss_index()
    if index is None:
        raise HTTPException(status_code=500, detail="No FAISS index found")

    # Fetch document metadata
    conn = get_db()
    docs_tbl, _ = get_or_create_tables(conn)
    all_docs = docs_tbl.search().to_list()
    doc_map = {d["doc_id"]: d for d in all_docs}

    # Tokenize question
    tokens = [tok for tok in re.findall(r"[a-zA-Z0-9']+", question.lower()) if tok]

    # Determine question category
    category = determine_question_category(question)

    # Run similarity search in FAISS
    retrieved = index.similarity_search(question, k=top_k * 5)

    if not retrieved:
        return {"answer": "I couldn't find relevant content.", "citations": []}

    # Identify docs whose keywords intersect with question tokens
    keyword_docs = set()
    for d in all_docs:
        kws = [kw.lower() for kw in (d.get("keywords") or [])]
        if any(tok in kws for tok in tokens):
            keyword_docs.add(d["doc_id"])

    # Rank results by lexical match, keyword match, and category
    candidates = []
    for idx, doc in enumerate(retrieved):
        meta = doc.metadata or {}
        text_lower = doc.page_content.lower()
        lex_match = 1 if any(tok in text_lower for tok in tokens) else 0
        doc_id = meta.get("doc_id")
        # Keyword score
        if doc_id in keyword_docs:
            kw_score = 10
        else:
            kws = [kw.lower() for kw in (doc_map.get(doc_id, {}).get("keywords") or [])]
            kw_score = sum(1 for kw in kws if kw in question.lower())
        # Category match
        cat_score = 1 if category and doc_map.get(doc_id, {}).get("category") == category else 0
        candidates.append((doc, lex_match, kw_score, cat_score, idx))

    candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))
    top_docs = [d for d, _, _, _, _ in candidates[:top_k]]

    # Build contexts
    contexts = []
    for d in top_docs:
        meta = d.metadata or {}
        contexts.append({
            "doc_id": meta.get("doc_id"),
            "page_start": meta.get("page_start", 1),
            "page_end": meta.get("page_end", 1),
            "text": d.page_content
        })

    # Generate answer
    answer = answer_with_context(question, contexts)

    # Build citations
    citations = []
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

@app.get("/documents")
def list_documents():
    conn = get_db()
    docs, _ = get_or_create_tables(conn)
    # LanceTable lacks a direct to_list method; use a search query
    # without filters to retrieve all rows instead.
    return {"documents": list(docs.search().to_list())}


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
    """
    Ingest a PDF into a FAISS index.
    - Parses the PDF with DoclingLoader.
    - Cleans proprietary/CUI notices from each page.
    - Splits into overlapping chunks.
    - Embeds each chunk using OllamaEmbeddings.
    - Adds the vectors and metadata to a FAISS index stored on disk.
    - Populates the documents table with high-level metadata, category and keywords.
    """
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save PDF to temp file for Docling
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

    # Clean each page’s content
    for doc in documents:
        doc.page_content = remove_noise(doc.page_content)

    # Compute SHA of original PDF
    sha = sha256_bytes(pdf_bytes)

    # Derive stable doc_id
    base_for_id = (source_url or sha)[:64]
    doc_id = doc_id or hashlib.sha1(base_for_id.encode("utf-8")).hexdigest()

    # Prepare split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    split_docs = splitter.split_documents(documents)

    if not split_docs:
        raise HTTPException(status_code=400, detail="No text extracted")

    # Texts and metadata for FAISS
    texts = []
    metas = []
    for doc_obj in split_docs:
        meta = doc_obj.metadata or {}
        p_start = int(meta.get("page", 1))
        p_end = p_start
        texts.append(doc_obj.page_content)
        metas.append({
            "doc_id": doc_id,
            "page_start": p_start,
            "page_end": p_end,
            "title": title,
            "source_url": source_url,
        })

    # Load or create the FAISS index and add new vectors
    index = get_faiss_index()
    if index is None:
        index = FAISS.from_texts(texts, embedder, metadatas=metas)
    else:
        index.add_texts(texts, metadatas=metas)
    # Persist to disk
    index.save_local(FAISS_INDEX_PATH)

    # Extract category and keywords using the LLM
    full_text = "\n\n".join([d.page_content for d in documents])
    category, keywords = extract_extra_metadata(full_text)

    # Document-level metadata stored in the existing LanceDB "documents" table
    conn = get_db()
    docs_tbl, _ = get_or_create_tables(conn)
    page_count = len(documents)
    docs_tbl.delete(f"doc_id == '{doc_id}'")
    docs_tbl.add([{
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
    }])

    return {"status": "ok", "doc_id": doc_id, "chunks": len(texts)}
