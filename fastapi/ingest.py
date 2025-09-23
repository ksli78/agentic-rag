from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import uuid
import asyncio

from config import settings
from db import get_or_create_tables
from models import IngestUrl, IngestJsonPayload, IngestResponse, Document, Chunk
from utils import sha256_bytes, embed_texts, split_markdown_blocks

# Docling
from docling.document_converter import DocumentConverter

ingest_router = APIRouter(tags=["ingest"])

docs_tbl, chunks_tbl = get_or_create_tables()

def _doc_exists(sha256: str) -> bool:
    return docs_tbl.search().where(f"file_sha256 = '{sha256}'").limit(1).to_list() != []

def _make_chunk_id() -> str:
    return str(uuid.uuid4())

MIN_CHARS = 20  # skip ultra-tiny fragments

async def _ingest_markdown(md_text: str, *, doc_id: str, title: Optional[str], source_url: Optional[str],
                           num_pages: int, category: Optional[str], keywords: Optional[List[str]]) -> int:
    # Chunk markdown
    raw_chunks = [c for c in split_markdown_blocks(md_text, settings.chunk_size) if c and len(c.strip()) >= MIN_CHARS]

    if not raw_chunks:
        return 0

    # Embed one-by-one (utils now does that)
    embeddings = await embed_texts(raw_chunks)

    # Sanity-align in case of any mismatch (shouldn’t happen now, but be safe)
    n = min(len(raw_chunks), len(embeddings))
    raw_chunks = raw_chunks[:n]
    embeddings = embeddings[:n]

    # Upsert document metadata
    docs_tbl.add([{
        "doc_id": doc_id, "title": title, "source_url": source_url,
        "file_sha256": None, "num_pages": num_pages,
        "category": category, "keywords": keywords
    }], mode="overwrite")

    # Write chunks
    rows = []
    for i in range(n):
        rows.append({
            "doc_id": doc_id,
            "chunk_id": _make_chunk_id(),
            "page_start": 1,
            "page_end": num_pages,
            "text": raw_chunks[i],
            "embedding": embeddings[i],
        })
    if rows:
        chunks_tbl.add(rows)
    return len(rows)

@ingest_router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),  # comma-separated
):
    data = await file.read()
    sha = sha256_bytes(data)
    if _doc_exists(sha):
        raise HTTPException(409, "Document already ingested (sha256 match)")

    # Save temp file for Docling
    tmp_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(data)

    # Convert to Markdown with Docling
    converter = DocumentConverter()
    result = converter.convert(tmp_path)
    md_text = result.document.export_to_markdown()
    num_pages = getattr(result.document, "page_count", None) or 1

    # Clean common banners (CUI/warnings) if present
    ban_phrases = ["CONTROLLED UNCLASSIFIED INFORMATION", "CUI", "WARNING"]
    lines = [ln for ln in md_text.splitlines() if not any(bp in ln.upper() for bp in ban_phrases)]
    md_text = "\n".join(lines).strip()

    # Persist Document row with sha
    final_doc_id = doc_id or str(uuid.uuid4())
    docs_tbl.add([{
        "doc_id": final_doc_id,
        "title": title or file.filename,
        "source_url": source_url,
        "file_sha256": sha,
        "num_pages": num_pages,
        "category": category,
        "keywords": [k.strip() for k in (keywords or "").split(",") if k.strip()] or None
    }])

    # Chunk + embed + store chunks
    chunks = await _ingest_markdown(
        md_text,
        doc_id=final_doc_id,
        title=title or file.filename,
        source_url=source_url,
        num_pages=num_pages,
        category=category,
        keywords=[k.strip() for k in (keywords or "").split(",") if k.strip()] or None
    )
    return IngestResponse(status="ok", doc_id=final_doc_id, chunks=chunks, category=category, keywords=[k.strip() for k in (keywords or "").split(",") if k.strip()] or None)


@ingest_router.post("/ingest/json", response_model=IngestResponse)
async def ingest_json(payload: IngestJsonPayload):
    # Direct text -> markdown (as-is)
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
