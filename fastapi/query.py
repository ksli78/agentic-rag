from fastapi import APIRouter
from typing import List, Dict
import numpy as np

from config import settings
from db import get_or_create_tables
from models import QueryPayload, QueryResponse, Citation
from utils import embed_texts, chat_complete, cosine_similarity


query_router = APIRouter(tags=["query"])

docs_tbl, chunks_tbl = get_or_create_tables()

SYSTEM_PROMPT = (
    "You are a helpful assistant for company policy Q&A. "
    "Use ONLY the provided CONTEXT to answer. "
    "Cite sources inline like [1], [2] etc. "
    "If the CONTEXT is insufficient, say you don't have the information and provide the portal link given."
)

def _format_context(chunks: List[Dict]) -> str:
    ctx_lines = []
    for idx, ch in enumerate(chunks, start=1):
        ctx_lines.append(f"[{idx}] (doc_id={ch['doc_id']}, p.{ch['page_start']}-{ch['page_end']})\n{ch['text']}\n")
    return "\n".join(ctx_lines)

def _citations_for(chunks: List[Dict]) -> List[Citation]:
    result: List[Citation] = []
    # fetch doc metadata for titles/urls
    doc_ids = list({ch["doc_id"] for ch in chunks})
    meta = {}
    if doc_ids:
        where = " OR ".join([f"doc_id = '{d}'" for d in doc_ids])
        for d in docs_tbl.search().where(where).to_list():
            meta[d["doc_id"]] = d
    for ch in chunks:
        d = meta.get(ch["doc_id"], {})
        result.append(Citation(
            doc_id=ch["doc_id"],
            title=d.get("title"),
            source_url=d.get("source_url"),
            page_start=int(ch["page_start"]),
            page_end=int(ch["page_end"]),
        ))
    return result

@query_router.post("/query", response_model=QueryResponse)
async def query(payload: QueryPayload) -> QueryResponse:
    q = payload.question.strip()
    if not q:
        return QueryResponse(answer="Please provide a question.", citations=[])

    # Embed question
    qvec = (await embed_texts([q]))[0]
    qvec_np = np.array(qvec, dtype=np.float32)

    # ANN search (use Lance index)
    # pull more initial candidates then filter hard
    candidate_k = max(payload.top_k * 6, 12)
    hits = chunks_tbl.search(qvec).limit(candidate_k).to_list()

    # Score + filter by similarity threshold
    scored = []
    for h in hits:
        sim = cosine_similarity(qvec_np, np.array(h["embedding"], dtype=np.float32))
        if sim >= settings.min_sim_threshold:
            h["score"] = sim
            scored.append(h)

    if not scored:
        fallback = (
            f"I’m sorry — I don’t have enough information in the current documents to answer that. "
            f"Please refer to the Space Missions Portal for details: {settings.portal_url}"
        )
        return QueryResponse(answer=fallback, citations=[])

    # Best chunk per doc
    best_per_doc: Dict[str, Dict] = {}
    for h in scored:
        d = h["doc_id"]
        if (d not in best_per_doc) or (h["score"] > best_per_doc[d]["score"]):
            best_per_doc[d] = h

    # Sort docs by score and keep top N
    top_docs = sorted(best_per_doc.values(), key=lambda x: x["score"], reverse=True)
    top_docs = top_docs[: min(payload.top_k, settings.max_context_docs)]

    # Build context and ask LLM
    context = _format_context(top_docs)
    user_prompt = (
        f"QUESTION:\n{q}\n\n"
        f"CONTEXT (citations in square brackets):\n{context}\n\n"
        "Answer succinctly and include the appropriate citation indices inline."
    )
    temperature = payload.temperature if payload.temperature is not None else settings.default_temperature
    answer = await chat_complete(SYSTEM_PROMPT, user_prompt, temperature=temperature)

    return QueryResponse(answer=answer, citations=_citations_for(top_docs))
