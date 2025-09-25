"""API endpoints for querying ingested documents.

This router exposes a single endpoint ``POST /query`` which accepts
a natural language question, computes its embedding via Ollama,
performs a hybrid dense/lexical search against the persistent index
store, formats the retrieved chunks into a context window, invokes
Ollama's chat API to generate an answer, and returns that answer
along with properly formatted citations.  If no relevant chunks are
found the service falls back to a portal URL specified in the
configuration.
"""

from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter

from config import settings
from models import QueryPayload, QueryResponse, Citation
from storage import store
from utils import embed_texts, chat_complete


query_router = APIRouter(tags=["query"])


# System prompt that constrains the behaviour of the assistant.
SYSTEM_PROMPT = (
    "You are a helpful assistant for company policy Q&A. "
    "Use ONLY the provided CONTEXT to answer. "
    "Cite sources inline like [1], [2] etc. "
    "If the CONTEXT is insufficient, say you don't have the information and provide the portal link given."
    "when referring to the CONTEXT call it the document or the information provided"
)


def _get_chunk_text(meta: Dict[str, any]) -> str:
    """Lookup the raw text for a chunk given its metadata."""
    cid = meta.get("chunk_id")
    # Find the index of the chunk_id in the metadata list
    for i, m in enumerate(store.chunk_metadata):
        if m.get("chunk_id") == cid:
            return store.chunk_texts[i]
    return ""


def _format_context(chunks: List[Dict[str, any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines: List[str] = []
    for idx, ch in enumerate(chunks, start=1):
        text = _get_chunk_text(ch)
        page_start = ch.get("page_start", 1)
        page_end = ch.get("page_end", 1)
        lines.append(
            f"[{idx}] (doc_id={ch['doc_id']}, p.{page_start}-{page_end})\n{text}\n"
        )
    return "\n".join(lines)


def _citations_for(chunks: List[Dict[str, any]]) -> List[Citation]:
    """Create a list of citation objects for the retrieved chunks."""
    doc_ids = list({ch["doc_id"] for ch in chunks})
    meta_lookup = store.get_document_metadata(doc_ids)
    citations: List[Citation] = []
    for ch in chunks:
        d = meta_lookup.get(ch["doc_id"], {})
        citations.append(
            Citation(
                doc_id=ch["doc_id"],
                title=d.get("title"),
                source_url=d.get("source_url"),
                page_start=int(ch.get("page_start", 1)),
                page_end=int(ch.get("page_end", 1)),
            )
        )
    return citations

@query_router.post("/query", response_model=QueryResponse)
async def query(payload: QueryPayload) -> QueryResponse:
    """Answer a question using hybrid search over ingested documents."""
    question = payload.question.strip()
    if not question:
        return QueryResponse(answer="Please provide a question.", citations=[])
    # Compute embedding for the query
    [qvec] = await embed_texts([question])
    # Hybrid search
    results = store.search(qvec, question, top_k=payload.top_k)
    if not results:
        fallback = (
            f"I'm sorry — I don't have enough information in the current documents to answer that. "
            f"Please refer to the Space Missions Portal for details: {settings.portal_url}"
        )
        return QueryResponse(answer=fallback, citations=[])
    # Group results by document and select up to ``settings.max_chunks_per_doc`` per document
    chunks_per_doc: Dict[str, List[Dict[str, any]]] = {}
    doc_order: List[str] = []
    for meta in results:
        d_id = meta["doc_id"]
        # Record the order in which documents appear
        if d_id not in doc_order:
            doc_order.append(d_id)
        # Append meta to the list for this document if under the limit
        if d_id not in chunks_per_doc:
            chunks_per_doc[d_id] = []
        if len(chunks_per_doc[d_id]) < settings.max_chunks_per_doc:
            chunks_per_doc[d_id].append(meta)
    # Limit number of documents considered based on top_k and max_context_docs
    selected_docs = doc_order[: min(payload.top_k, settings.max_context_docs)]
    # Flatten selected chunks in document order
    selected_chunks: List[Dict[str, any]] = []
    for d_id in selected_docs:
        selected_chunks.extend(chunks_per_doc.get(d_id, []))
    # Build context string
    context = _format_context(selected_chunks)
    # Compose user prompt for chat model
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT (citations in square brackets):\n{context}\n\n"
        "Answer with a detailed summary and include the appropriate citation indices inline."
        "make sure the answer matches the question, for example if the user askes about PTO (Paid Time Off) dont add any information about Telecommunting"
        "even if it is in the context"
        "format your answer with PROPER HTML TAGS  us UL, OL, TABLE,P , H and DIV TAGS"
    )
    # Temperature
    temperature = payload.temperature if payload.temperature is not None else settings.default_temperature
    # Generate answer using Ollama chat API
    answer = await chat_complete(SYSTEM_PROMPT, user_prompt, temperature=temperature)
    # Format citations
    citations = _citations_for(selected_chunks)
    return QueryResponse(answer=answer, citations=citations)
