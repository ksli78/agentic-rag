"""API endpoints for document ingestion.

This router defines a single endpoint ``POST /ingest/upload`` which
accepts a PDF file upload, extracts its text using Docling (or
PyMuPDF as a fallback), splits the text into manageable chunks,
computes dense embeddings via Ollama, and stores both the dense and
lexical representations in the persistent index.  Duplicate uploads
are detected using the file's SHA256 hash.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from config import settings
from models import IngestResponse
from storage import store
from utils import sha256_bytes, embed_texts, split_markdown_blocks
from utils import extract_category_keywords  # new import

log = logging.getLogger("api.ingest")

ingest_router = APIRouter(tags=["ingest"])

# Parser selection
ENGINE = settings.doc_parse_engine.lower()
try:
    from docling.document_converter import DocumentConverter  # type: ignore
except Exception:
    DocumentConverter = None


def extract_pdf_to_markdown(tmp_path: str) -> tuple[str, int]:
    """Extract Markdown text and page count from a PDF.

    If ``settings.doc_parse_engine`` is ``"docling"`` and Docling is
    available, use it to convert the PDF to Markdown.  Otherwise use
    PyMuPDF as a fallback.  Returns a tuple ``(markdown_text, num_pages)``.
    """
    if ENGINE == "docling" and DocumentConverter is not None:
        log.info("Docling: converting PDF -> Markdown ...")
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        md_text = result.document.export_to_markdown()
        num_pages = getattr(result.document, "page_count", None) or 1
        return md_text, num_pages
    # Fallback: PyMuPDF (fitz)
    import fitz  # type: ignore
    log.info("PyMuPDF: converting PDF -> Markdown ...")
    doc = fitz.open(tmp_path)
    parts: List[str] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("markdown") or page.get_text() or ""
        parts.append(f"\n\n## Page {i}\n\n{text.strip()}")
    md_text = "\n".join(parts).strip()
    return md_text, len(doc)


MIN_CHARS = 20  # drop tiny fragments


@ingest_router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
) -> IngestResponse:
    """Ingest a PDF document into the RAG system.

    The uploaded file is converted to Markdown, chunked, embedded and
    stored.  Duplicate detection is performed using the file's
    SHA256 hash.  Returns metadata about the ingestion.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    # Read file contents
    data = await file.read()
    sha = sha256_bytes(data)
    # Check for duplicate (by SHA)
    for doc in store.documents:
        if doc.get("file_sha256") == sha:
            raise HTTPException(409, "Document already ingested (sha256 match).")
    # Write temp file
    tmp_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(data)
    # Convert to Markdown
    md_text, num_pages = extract_pdf_to_markdown(tmp_path)
    log.info(f"Converted PDF '{file.filename}' -> Markdown. pages={num_pages}")
    # Clean up temporary file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    # Remove common banners
    ban_phrases = ["CONTROLLED UNCLASSIFIED INFORMATION", "CUI", "WARNING"]
    lines = [ln for ln in md_text.splitlines() if not any(bp in ln.upper() for bp in ban_phrases)]
    md_text = "\n".join(lines).strip()
    # Split into blocks
    raw_chunks = [c for c in split_markdown_blocks(md_text, settings.chunk_size) if c and len(c.strip()) >= MIN_CHARS]
    log.info(f"Chunked into {len(raw_chunks)} blocks (>= {MIN_CHARS} chars).")
    if not raw_chunks:
        log.warning("No usable text/chunks produced from PDF. Skipping ingestion.")
        return IngestResponse(status="no_chunks", doc_id=doc_id or "", chunks=0, category=category, keywords=None)
    # Compute embeddings
    embeddings = await embed_texts(raw_chunks)
    if len(embeddings) != len(raw_chunks):
        log.error(f"Embedding count mismatch: chunks={len(raw_chunks)} embeddings={len(embeddings)}")
        raise HTTPException(status_code=500, detail="Embedding mismatch; check Ollama embeddings endpoint.")
    
    # Determine final document ID
    final_doc_id = doc_id or str(uuid.uuid4())
    kw_list = [k.strip() for k in (keywords or "").split(",") if k.strip()] or None

    # Auto-generate category and keywords via LLM if not provided
    if (not category) or (not kw_list):
        # Use a portion of the original Markdown text for extraction
        doc_excerpt = md_text[:4096]
        cat_pred, kw_pred = await extract_category_keywords(doc_excerpt)
        if not category and cat_pred:
            category = cat_pred
        if not kw_list and kw_pred:
            kw_list = kw_pred

    # Build document metadata with the possibly updated category/keywords
    doc_meta = {
        "doc_id": final_doc_id,
        "title": title or file.filename,
        "source_url": source_url,
        "file_sha256": sha,
        "num_pages": num_pages,
        "category": category,
        "keywords": kw_list,
    }

    # Build chunk metadata
    chunk_meta: List[dict] = []
    for _ in range(len(raw_chunks)):
        meta = {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": final_doc_id,
            "title": doc_meta["title"],
            "source_url": doc_meta["source_url"],
            "page_start": 1,
            "page_end": num_pages,
        }
        chunk_meta.append(meta)
    # Store in index
    store.add_documents(embeddings, raw_chunks, chunk_meta, doc_meta)
    return IngestResponse(status="ok", doc_id=final_doc_id, chunks=len(raw_chunks), category=category, keywords=kw_list)
