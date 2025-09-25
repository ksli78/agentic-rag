"""Main application setup for the Agentic RAG service.

This module constructs the FastAPI application, configures CORS, mounts
the ingestion and query routers, and exposes utility endpoints for
health checks, listing documents, retrieving a single document with its
chunks, and erasing all stored data.  All stateful storage is
concentrated in the ``IndexStore`` instance exposed in ``storage``.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from ingest import ingest_router
from query import query_router
from storage import store
from models import Document, Chunk, DocumentsResponse, DocumentWithChunksResponse


# Configure logging according to settings
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

app = FastAPI(title="Agentic-RAG (Docling Refactor)", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health")
def health() -> dict:
    """Return a simple health status."""
    return {"status": "ok"}


@app.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    """List all ingested documents."""
    docs: List[Document] = [Document(**d) for d in store.documents]
    return DocumentsResponse(documents=docs)


@app.get("/document/{doc_id}", response_model=DocumentWithChunksResponse)
def get_document(doc_id: str) -> DocumentWithChunksResponse:
    """Retrieve a single document and its chunks by ID."""
    meta = None
    for d in store.documents:
        if d.get("doc_id") == doc_id:
            meta = d
            break
    if meta is None:
        raise HTTPException(404, "Document not found")
    # Gather chunk objects for the document
    chunk_objs: List[Chunk] = []
    for m, text in zip(store.chunk_metadata, store.chunk_texts):
        if m.get("doc_id") == doc_id:
            chunk_objs.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=m.get("chunk_id"),
                    page_start=int(m.get("page_start", 1)),
                    page_end=int(m.get("page_end", 1)),
                    text=text,
                    embedding=[],
                )
            )
    return DocumentWithChunksResponse(document=Document(**meta), chunks=chunk_objs)


@app.post("/erase")
def erase_all() -> dict:
    """Erase all stored documents and chunks."""
    # Remove the storage directory and recreate a fresh store
    dir_path = settings.faiss_dir
    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)
    # Reinitialise the global store
    from storage import store as global_store  # type: ignore
    # Force reload by reassigning attributes (can't reassign module variable here)
    global_store.documents.clear()
    global_store.chunk_metadata.clear()
    global_store.chunk_texts.clear()
    # Destroy and recreate FAISS index
    import faiss  # type: ignore
    global_store.faiss_index = faiss.IndexFlatIP(settings.embed_dim)
    # Remove TF‑IDF model
    global_store.tfidf_vectorizer = None
    global_store.tfidf_matrix = None
    # Persist empty index
    global_store.save()
    return {"status": "erased"}
