# fastapi/app.py
import logging, sys
from config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ingest import ingest_router
from query import query_router
from db import get_or_create_tables
from models import DocumentsResponse, DocumentWithChunksResponse

app = FastAPI(title="Agentic-RAG (Docling Refactor)", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

app.include_router(ingest_router)
app.include_router(query_router)

docs_tbl, chunks_tbl = get_or_create_tables()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    rows = docs_tbl.search().to_list()
    return DocumentsResponse(documents=rows)

@app.get("/document/{doc_id}", response_model=DocumentWithChunksResponse)
def get_document(doc_id: str):
    docs = docs_tbl.search().where(f"doc_id = '{doc_id}'").limit(1).to_list()
    if not docs:
        raise HTTPException(404, "Document not found")
    ch = chunks_tbl.search().where(f"doc_id = '{doc_id}'").to_list()
    return DocumentWithChunksResponse(document=docs[0], chunks=ch)

@app.post("/erase")
def erase_all():
    docs_tbl.delete("true")
    chunks_tbl.delete("true")
    return {"status": "erased"}
