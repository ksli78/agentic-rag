# fastapi/ingest.py
import os
import uuid
import logging
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from config import settings
from db import get_or_create_tables
from models import IngestJsonPayload, IngestResponse
from utils import sha256_bytes, embed_texts, split_markdown_blocks

log = logging.getLogger("api.ingest")
ingest_router = APIRouter(tags=["ingest"])
docs_tbl, chunks_tbl = get_or_create_tables()

# Parser selection (docling | fitz)
ENGINE = settings.doc_parse_engine.lower()
try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None

def extract_pdf_to_markdown(tmp_path: str) -> tuple[str, int]:
    """Return (markdown_text, num_pages) using selected engine."""
    if ENGINE == "docling" and DocumentConverter is not None:
        log.info("Docling: converting PDF -> Markdown ...")
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        md_text = result.document.export_to_markdown()
        num_pages = getattr(result.document, "page_count", None) or 1
        return md_text, num_pages

    # Fallback: PyMuPDF (fitz)
    import fitz
    log.info("PyMuPDF: converting PDF -> Markdown ...")
    doc = fitz.open(tmp_path)
    parts = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("markdown") or page.get_text() or ""
        parts.append(f"\n\n## Page {i}\n\n{text.strip()}")
    md_text = "\n".join(parts).strip()
    return md_text, len(doc)

MIN_CHARS = 20  # drop tiny fragments

async def _ingest_markdown(md_text: str, *, doc_id: str, title: Optional[str],
                           source_url: Optional[str], num_pages: int,
                           category: Optional[str], keywords: Optional[List[str]]) -> int:
    log.info(f"Markdown length={len(md_text)}. Preview:\n{(md_text[:500] + '...') if len(md_text) > 500 else md_text}")

    # Remove common banners
    ban_phrases = ["CONTROLLED UNCLASSIFIED INFORMATION", "CUI", "WARNING"]
    lines = [ln for ln in md_text.splitlines() if not any(bp in ln.upper() for bp in ban_phrases)]
    md_text = "\n".join(lines).strip()

    raw_chunks = [c for c in split_markdown_blocks(md_text, settings.chunk_size) if c and len(c.strip()) >= MIN_CHARS]
    log.info(f"Chunked into {len(raw_chunks)} blocks (>= {MIN_CHARS} chars).")

    if not raw_chunks:
        log.warning("No usable text/chunks produced from PDF. Skipping ingestion.")
        return 0

    # Embeddings
    embeddings = await embed_texts(raw_chunks)
    if len(embeddings) != len(raw_chunks):
        log.error(f"Embedding count mismatch: chunks={len(raw_chunks)} embeddings={len(embeddings)}")
        raise HTTPException(status_code=500, detail="Embedding mismatch (check Ollama embeddings endpoint).")

    for idx, vec in enumerate(embeddings):
        if not vec:
            log.error(f"Empty embedding at chunk {idx}.")
            raise HTTPException(status_code=500, detail=f"Empty embedding at chunk {idx}.")
        if len(vec) != settings.embed_dim:
            log.error(f"Embedding dim mismatch at chunk {idx}: got {len(vec)} expected {settings.embed_dim}")
            raise HTTPException(status_code=500, detail="Embedding dim mismatch; set EMBED_DIM correctly.")

    # Upsert metadata
    docs_tbl.add([{
        "doc_id": doc_id,
        "title": title,
        "source_url": source_url,
        "file_sha256": None,
        "num_pages": num_pages,
        "category": category,
        "keywords": keywords
    }], mode="overwrite")

    # Write chunks
    rows = []
    for i, text in enumerate(raw_chunks):
        rows.append({
            "doc_id": doc_id,
            "chunk_id": str(uuid.uuid4()),
            "page_start": 1,
            "page_end": num_pages,
            "text": text,
            "embedding": embeddings[i],
        })
    chunks_tbl.add(rows)
    log.info(f"Wrote {len(rows)} chunks for doc_id={doc_id}")
    return len(rows)

@ingest_router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
):
    data = await file.read()
    sha = sha256_bytes(data)

    # Duplicate detection (by sha)
    existing = docs_tbl.search().where(f"file_sha256 = '{sha}'").limit(1).to_list()
    if existing:
        raise HTTPException(409, "Document already ingested (sha256 match).")

    tmp_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(data)

    md_text, num_pages = extract_pdf_to_markdown(tmp_path)
    log.info(f"Converted PDF '{file.filename}' -> Markdown. pages={num_pages}")

    final_doc_id = doc_id or str(uuid.uuid4())
    kw_list = [k.strip() for k in (keywords or "").split(",") if k.strip()] or None

    # Store doc metadata with sha (so duplicates are caught later)
    docs_tbl.add([{
        "doc_id": final_doc_id,
        "title": title or file.filename,
        "source_url": source_url,
        "file_sha256": sha,
        "num_pages": num_pages,
        "category": category,
        "keywords": kw_list
    }])

    chunks = await _ingest_markdown(
        md_text,
        doc_id=final_doc_id,
        title=title or file.filename,
        source_url=source_url,
        num_pages=num_pages,
        category=category,
        keywords=kw_list
    )
    return IngestResponse(status="ok", doc_id=final_doc_id, chunks=chunks, category=category, keywords=kw_list)

@ingest_router.post("/ingest/json", response_model=IngestResponse)
async def ingest_json(payload: IngestJsonPayload):
    chunks = await _ingest_markdown(
        payload.text,
        doc_id=payload.doc_id,
        title=payload.title,
        source_url=payload.source_url,
        num_pages=payload.num_pages or 1,
        category=payload.category,
        keywords=payload.keywords,
    )
    return IngestResponse(status="ok", doc_id=payload.doc_id, chunks=chunks, category=payload.category, keywords=payload.keywords)
