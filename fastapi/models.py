"""Pydantic models for API request/response payloads and data storage.

These models define the schemas used throughout the FastAPI service.  They
are plain Pydantic ``BaseModel`` subclasses with type hints and
validation.  The ``Document`` and ``Chunk`` models correspond to
metadata about uploaded documents and their extracted chunks.  Other
models encapsulate API requests and responses.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Metadata for an ingested document."""
    doc_id: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    file_sha256: Optional[str] = None
    num_pages: Optional[int] = None
    category: Optional[str] = None
    keywords: Optional[List[str]] = None


class Chunk(BaseModel):
    """Represents a contiguous block of text extracted from a document."""
    doc_id: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    embedding: List[float]


class Citation(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    page_start: int
    page_end: int


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    chunks: int
    category: Optional[str] = None
    keywords: Optional[List[str]] = None


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
