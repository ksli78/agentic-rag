# fastapi/models.py
from typing import List, Optional
from pydantic import BaseModel, Field
from lancedb.pydantic import LanceModel, Vector
from config import settings

# ---------- LanceDB table schemas ----------

class Document(LanceModel):
    doc_id: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    file_sha256: Optional[str] = None
    num_pages: Optional[int] = None
    category: Optional[str] = None
    keywords: Optional[List[str]] = None

class Chunk(LanceModel):
    doc_id: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    embedding: Vector(settings.embed_dim)  # ← uses EMBED_DIM from .env

# ---------- API request/response models ----------

class IngestJsonPayload(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    text: str
    num_pages: Optional[int] = 1
    category: Optional[str] = None
    keywords: Optional[List[str]] = None

class IngestResponse(BaseModel):
    status: str
    doc_id: str
    chunks: int
    category: Optional[str] = None
    keywords: Optional[List[str]] = None

class DeleteDocumentPayload(BaseModel):
    doc_id: str

class DeleteResponse(BaseModel):
    status: str
    doc_id: str

class Citation(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    page_start: int
    page_end: int

class QueryPayload(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]

class DocumentsResponse(BaseModel):
    documents: List[Document]

class DocumentWithChunksResponse(BaseModel):
    document: Document
    chunks: List[Chunk]
