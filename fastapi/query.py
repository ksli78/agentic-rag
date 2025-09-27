from __future__ import annotations
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
import re
import asyncio
import logging


from typing import Dict, List

from fastapi import APIRouter

from config import settings
from models import QueryPayload, QueryResponse, Citation
from storage import store
from utils import embed_texts, chat_complete
try:
    from sentence_transformers import CrossEncoder
    # Load the lightweight MS MARCO re-ranker on startup
    _RE_RANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as ex:
    logging.warning(
        "CrossEncoder re-ranker could not be loaded. "
        "Falling back to simple hybrid ranking. "
        f"Error: {ex}"
    )
    _RE_RANKER = None

query_router = APIRouter(tags=["query"])

# A minimal set of English stopwords
COMMON_STOPWORDS = {
    "what", "is", "the", "a", "an", "and", "or", "to", "of", "in", "on",
    "for", "by", "with", "as", "at", "from", "into", "about", "it",
    # Add a few too-generic tokens that caused false positives in PTO:
    "time", "off"
}
# Terms that are too general to be useful for filtering.
DOMAIN_STOPWORDS = {
   "nasa", "space", "mission", "division", "asmd", "amentum", "rocket",
   "launch", "satellite", "astronaut", "flight", "engineering", "technical",
   "safety", "quality", "assurance", "control", "project", "program", "management",
   "reporting", "contract", "federal", "government", "compliance", "regulation",
   "standard", "requirement", "data", "system", "infrastructure", "operations",
   "maintenance", "testing", "development", "research", "team", "personnel",
   "training", "security", "facility", "asset", "material", "supply", "chain",
   "procurement", "finance", "budget", "billing", "travel", "communication", "meeting",
   "review", "proposal", "customer", "partner", "performance", "metric", "objective",
   "goal", "strategy", "innovation", "technology", "software", "hardware", "tool",
   "specification", "analysis", "risk", "mitigation", "hazard", "incident", "investigation",
   "manual", "guide", "record", "archive", "timeline", "milestone", "delivery",
   "integration", "validation", "verification", "support", "planning", "future",
   "teammate", "policy", "procedure", "process", "document",
   "employee", "employees", "company", "department",
   "section", "subsection"
}

# Build a mapping of acronyms to their full-word tokens, e.g. {'pto': ['paid','time','off']}
SYNONYMS_MAP: dict[str, list[str]] = {}
# Regex to match phrases like "Paid Time Off (PTO)" or "Paid Time Off – PTO"
_ACRONYM_REGEX = re.compile(
    r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+)\s*[\(\-–—]\s*([A-Z]{2,})\s*[\)\-–—]'
)


# System prompt that constrains the behaviour of the assistant.
SYSTEM_PROMPT = (
    "You are a helpful assistant for company policy Q&A. "
    "Use ONLY the provided CONTEXT to answer. "
    "Cite sources inline like [1], [2] etc. "
    "If the CONTEXT is insufficient, say you don't have the information and provide the portal link given."
    "when referring to the CONTEXT call it the document or the information provided"
)
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())

def extend_chunks(chunks: List[Dict[str, any]], max_extra: int = 2) -> List[Dict[str, any]]:
    """
    Append up to `max_extra` subsequent chunks from the same document
    for each item in `chunks`. Avoid duplicates. This helps include
    full bullet lists or paragraph continuations.
    """
    extended: List[Dict[str, any]] = []
    seen: set[str] = set()

    # Build index lookup for speed
    index_by_id = {m["chunk_id"]: idx for idx, m in enumerate(store.chunk_metadata)}

    for meta in chunks:
        cid = meta["chunk_id"]
        doc_id = meta["doc_id"]

        if cid not in seen:
            extended.append(meta)
            seen.add(cid)

        # add up to N subsequent chunks from the same doc
        if cid in index_by_id:
            idx = index_by_id[cid]
            # walk forward
            extras_added = 0
            j = idx + 1
            while j < len(store.chunk_metadata) and extras_added < max_extra:
                nxt = store.chunk_metadata[j]
                if nxt["doc_id"] != doc_id:
                    break
                if nxt["chunk_id"] not in seen:
                    extended.append(nxt)
                    seen.add(nxt["chunk_id"])
                    extras_added += 1
                j += 1

    return extended


def extract_keywords(text: str) -> list[str]:
    """Remove common and domain stopwords from tokens."""
    return [
        tok for tok in tokenize(text)
        if tok not in COMMON_STOPWORDS and tok not in DOMAIN_STOPWORDS
    ]

def has_keyword_overlap(query_keywords: list[str], chunk_text: str) -> bool:
    chunk_keywords = set(extract_keywords(chunk_text))
    return any(keyword in chunk_keywords for keyword in query_keywords)

def extract_query_tokens(question: str) -> list[str]:
    """
    Split the question into lowercase alphanumeric tokens,
    removing generic stopwords.
    """
    tokens = re.findall(r"[a-zA-Z0-9']+", question.lower())
    return [tok for tok in tokens if tok not in DOMAIN_STOPWORDS]

def chunk_contains_token(meta: dict, tokens: list[str], get_chunk_text) -> bool:
    """
    Return True if the chunk's text contains any of the given tokens.
    """
    text = get_chunk_text(meta).lower()
    return any(tok in text for tok in tokens)

def _get_chunk_text(meta: Dict[str, any]) -> str:
    """Lookup the raw text for a chunk given its metadata."""
    cid = meta.get("chunk_id")
    for i, m in enumerate(store.chunk_metadata):
        if m.get("chunk_id") == cid:
            return store.chunk_texts[i]
    return ""

def _format_context(chunks: List[Dict[str, any]]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Use document TITLE rather than internal doc_id to keep human-readable.
    """
    # Build title lookup
    doc_ids = list({ch["doc_id"] for ch in chunks})
    meta_lookup = store.get_document_metadata(doc_ids)  # doc_id -> {title, ...}
    lines: List[str] = []
    for idx, ch in enumerate(chunks, start=1):
        title = meta_lookup.get(ch["doc_id"], {}).get("title") or ch.get("title") or "Document"
        page_start = ch.get("page_start", 1)
        page_end = ch.get("page_end", 1)
        text = _get_chunk_text(ch)
        lines.append(f"[{idx}] ({title}, p.{page_start}-{page_end})\n{text}\n")
    return "\n".join(lines)


def _citations_for(chunks: List[Dict[str, any]]) -> List[Citation]:
    """
    Build citations with human-friendly titles. Titles are taken from document metadata.
    """
    doc_ids = list({ch["doc_id"] for ch in chunks})
    meta_lookup = store.get_document_metadata(doc_ids)
    citations: List[Citation] = []
    for ch in chunks:
        d = meta_lookup.get(ch["doc_id"], {})
        citations.append(
            Citation(
                doc_id=ch["doc_id"],
                title=d.get("title") or ch.get("title"),
                source_url=d.get("source_url"),
                page_start=int(ch.get("page_start", 1)),
                page_end=int(ch.get("page_end", 1)),
            )
        )
    return citations

def _build_synonyms_map():
    global SYNONYMS_MAP
    for text in store.chunk_texts:
        for phrase, acronym in _ACRONYM_REGEX.findall(text):
            # keep only meaningful tokens from phrase
            tokens = [
                tok for tok in tokenize(phrase)
                if tok not in COMMON_STOPWORDS and tok not in DOMAIN_STOPWORDS
            ]
            if tokens:
                SYNONYMS_MAP[acronym.lower()] = tokens

# Call this once when FastAPI starts
_build_synonyms_map()

@query_router.post("/query", response_model=QueryResponse)
async def query(payload: QueryPayload) -> QueryResponse:
    """
    Answer a question using hybrid search over ingested documents and a cross-encoder
    re-ranker for improved relevance.  Keywords in the question (including acronym
    expansions) are used to bias the final ranking so that passages containing the
    query terms are prioritized.
    """
    question = payload.question.strip()
    if not question:
        return QueryResponse(answer="Please provide a question.", citations=[])

    # 1) Dense embedding for the query
    [qvec] = await embed_texts([question])

    # 2) Initial hybrid retrieval (dense + TF-IDF), larger pool
    candidate_pool = payload.top_k * 10
    results = store.search(qvec, question, top_k=candidate_pool)

    # 3) Extract and expand query keywords (handle acronyms like PTO -> ['paid','time','off'])
    query_keywords = extract_keywords(question)
    expanded = set(query_keywords)
    for kw in query_keywords:
        if kw in SYNONYMS_MAP:
            expanded.update(SYNONYMS_MAP[kw])
    query_keywords = list(expanded)

    # 4) Lexical filter: Keep only chunks with overlap in keywords
    filtered = [
        meta for meta in results
        if has_keyword_overlap(query_keywords, _get_chunk_text(meta))
    ]
    if filtered:
        results = filtered

    # 5) Neighbor inclusion: include next 2 chunks per result to capture continuation
    results = extend_chunks(results, max_extra=2)

    # 6) Re-rank via cross-encoder (if available), then keyword-weight the scores
    if _RE_RANKER and results:
        pairs = [(question, _get_chunk_text(meta)) for meta in results]
        scores = await asyncio.get_event_loop().run_in_executor(None, _RE_RANKER.predict, pairs)

        # Soft keyword boost (1.5x if keyword present)
        keyword_weights = []
        for meta in results:
            text = _get_chunk_text(meta).lower()
            ratio = 1.0 if any(kw in text for kw in query_keywords) else 0.0
            keyword_weights.append(1.0 + 0.5 * ratio)  # 1.5x max

        weighted_scores = [s * w for s, w in zip(scores, keyword_weights)]
        order = sorted(zip(results, weighted_scores), key=lambda x: x[1], reverse=True)
        results = [meta for meta, _ in order]

    # 7) If nothing left after filters/ranking, return fallback
    if not results:
        fallback = (
            "I'm sorry — I don't have enough information in the current documents to "
            "answer that. Please refer to the Space Missions Portal for details: "
            f"{settings.portal_url}"
        )
        return QueryResponse(answer=fallback, citations=[])

    # 8) Group by document and cap chunks per doc to avoid flooding
    chunks_per_doc: Dict[str, List[Dict[str, any]]] = {}
    doc_order: List[str] = []
    for meta in results:
        d_id = meta["doc_id"]
        if d_id not in doc_order:
            doc_order.append(d_id)
        if d_id not in chunks_per_doc:
            chunks_per_doc[d_id] = []
        if len(chunks_per_doc[d_id]) < settings.max_chunks_per_doc:
            chunks_per_doc[d_id].append(meta)

    # 9) Cap number of documents used
    selected_docs = doc_order[: min(payload.top_k, settings.max_context_docs)]

    # 10) Flatten chunks in document order
    selected_chunks: List[Dict[str, any]] = []
    for d_id in selected_docs:
        selected_chunks.extend(chunks_per_doc.get(d_id, []))

    # 11) Format context (using TITLE instead of doc_id) and assemble the prompt
    context = _format_context(selected_chunks)

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT (citations in square brackets):\n{context}\n\n"
        "Answer succinctly and include the appropriate citation indices inline. "
        "Make sure the answer matches the question. "
        "Format your answer with PROPER HTML TAGS (UL, OL, TABLE, P, H, DIV)."
    )

    temperature = payload.temperature if payload.temperature is not None else settings.default_temperature

    # 12) Generate answer
    answer = await chat_complete(SYSTEM_PROMPT, user_prompt, temperature=temperature)

    # 13) Return answer + human-friendly citations
    citations = _citations_for(selected_chunks)
    return QueryResponse(answer=answer, citations=citations)

