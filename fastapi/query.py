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
}
# Terms that are too general to be useful for filtering.
DOMAIN_STOPWORDS = {
   "NASA", "space", "mission", "division", "ASMD", "Amentum", "rocket",
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
   "integration", "testing", "validation", "verification", "support", "planning", "strategy", "future",
   "teammate", "policy", "procedure", "process", "document",
    "employee", "employees", "company", "department",
    "division", "section", "subsection",
}

# Build a mapping of acronyms to their full-word tokens, e.g. {'pto': ['paid','time','off']}
SYNONYMS_MAP: dict[str, list[str]] = {}
# Regex to match phrases like "Paid Time Off (PTO)" or "Paid Time Off – PTO"
_ACRONYM_REGEX = re.compile(
    r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+)\s*'       # phrase with capitalised words
    r'[\(\-–—]\s*'                                    # left delimiter (parenthesis or dash)
    r'([A-Z]{2,})\s*'                                  # acronym (2+ uppercase letters)
    r'[\)\-–—]'                                        # right delimiter
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
    """Split text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9']+", text.lower())
def extend_chunks(chunks, max_extra=1):
    """
    Given a list of ranked chunk metadata, append up to `max_extra` subsequent
    chunks from the same document for each chunk.  Avoid duplicates.
    """
    extended = []
    seen = set()
    for meta in chunks:
        cid = meta["chunk_id"]
        doc_id = meta["doc_id"]
        # always keep the original chunk
        if cid not in seen:
            extended.append(meta)
            seen.add(cid)
        # find the next `max_extra` chunks in the same document
        extras_added = 0
        for m in store.chunk_metadata:
            if m["doc_id"] == doc_id and m["chunk_id"] > cid:
                if m["chunk_id"] not in seen:
                    extended.append(m)
                    seen.add(m["chunk_id"])
                    extras_added += 1
                    if extras_added >= max_extra:
                        break
    return extended

def extract_keywords(text: str) -> list[str]:
    """Remove common and domain stopwords from tokens."""
    return [
        tok
        for tok in tokenize(text)
        if tok not in COMMON_STOPWORDS and tok not in DOMAIN_STOPWORDS
    ]

def has_keyword_overlap(query_keywords: list[str], chunk_text: str) -> bool:
    """
    Return True if the chunk contains any keyword from the query.
    """
    chunk_keywords = set(extract_keywords(chunk_text))
    for keyword in query_keywords:
        if keyword in chunk_keywords:
            return True
    return False
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


def _build_synonyms_map():
    """Extract acronym mappings from the corpus."""
    global SYNONYMS_MAP
    for text in store.chunk_texts:
        for phrase, acronym in _ACRONYM_REGEX.findall(text):
            tokens = [
                tok.lower() for tok in phrase.split()
                if tok.lower() not in COMMON_STOPWORDS
                and tok.lower() not in DOMAIN_STOPWORDS
            ]
            if tokens:
                SYNONYMS_MAP[acronym.lower()] = tokens

# Call this once when FastAPI starts
_build_synonyms_map()

@query_router.post("/query", response_model=QueryResponse)
async def query(payload: QueryPayload) -> QueryResponse:
    """
    Answer a question using hybrid search over ingested documents and a cross-encoder
    re-ranker for improved relevance.  Keywords in the question are used to bias
    the final ranking so that passages containing the query terms are prioritized.
    """
    question = payload.question.strip()
    if not question:
        return QueryResponse(answer="Please provide a question.", citations=[])

    # Compute embedding for the query (dense vector)
    [qvec] = await embed_texts([question])

    # Use a larger pool for initial hybrid search to give the re-ranker more options.
    # Here we multiply the requested top_k by 10; adjust as needed.
    candidate_pool = payload.top_k * 10
    # Perform the hybrid (dense + lexical) search.
    results = store.search(qvec, question, top_k=candidate_pool)

    # Extract meaningful keywords from the question
    query_keywords = extract_keywords(question)

    # Expand keywords using the synonyms map (if any matches)
    expanded = set(query_keywords)
    
    for kw in query_keywords:
        if kw in SYNONYMS_MAP:
            # add all the words that define the acronym to the search keywords
            expanded.update(SYNONYMS_MAP[kw])
    # Use the expanded keyword set
    query_keywords = list(expanded)


    # Filter out candidate chunks that share no keywords with the query
    filtered = [
        meta
        for meta in results
        if has_keyword_overlap(query_keywords, _get_chunk_text(meta))
    ]
    # Use the filtered set if it’s non-empty; otherwise fall back to all results
    if filtered:
        results = filtered

    # after results have been re-ranked by the cross-encoder
    # extend the top results to include the next chunk from each doc
    extended_results = extend_chunks(results, max_extra=1)
    results = extended_results

    # If the re-ranker is loaded, use it to score each candidate chunk.
    # This step will reorder 'results' so that the most relevant passages
    # (according to the cross-encoder) appear first.
    if _RE_RANKER and results:
        # Build pairs of (question, chunk_text) for the re-ranker
        pairs = [
            (question, _get_chunk_text(meta))
            for meta in results
        ]
        # Run the cross-encoder in a background thread to avoid blocking the event loop
        scores = await asyncio.get_event_loop().run_in_executor(
            None,
            _RE_RANKER.predict,
            pairs
        )
        # ---------------------------------------
        # Keyword-weighted re-ranking section:
        # Boost the score of any chunk containing a query keyword.
        keyword_weights = []
        for meta in results:
            text = _get_chunk_text(meta).lower()
            # If any query keyword appears in the text, ratio=1.0; otherwise 0.0
            ratio = 1.0 if any(kw in text for kw in query_keywords) else 0.0
            # Weight = 1.0 + ratio (doubling the score for keyword-containing chunks)
            keyword_weights.append(1.0 + ratio)
        # Multiply cross-encoder scores by weights
        weighted_scores = [s * w for s, w in zip(scores, keyword_weights)]
        # Sort by weighted_scores descending
        scored_results = sorted(
            zip(results, weighted_scores),
            key=lambda x: x[1],
            reverse=True
        )
        # Update results to the weighted re-ranked order
        results = [meta for meta, _ in scored_results]
        # ---------------------------------------

    # If no results, fall back to portal link
    if not results:
        fallback = (
            "I'm sorry — I don't have enough information in the current documents to "
            "answer that. Please refer to the Space Missions Portal for details: "
            f"{settings.portal_url}"
        )
        return QueryResponse(answer=fallback, citations=[])

    # Group results by document and limit the number of chunks per document
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

    # Limit the number of documents based on top_k and max_context_docs
    selected_docs = doc_order[:min(payload.top_k, settings.max_context_docs)]

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
        "Answer succinctly and include the appropriate citation indices inline."
        "make sure the answer matches the question"
        "format your answer with PROPER HTML TAGS  us UL, OL, TABLE,P , H and DIV TAGS"
    )

    # Determine temperature
    temperature = (
        payload.temperature
        if payload.temperature is not None
        else settings.default_temperature
    )

    # Generate answer using Ollama chat API
    answer = await chat_complete(SYSTEM_PROMPT, user_prompt, temperature=temperature)

    # Build citations list
    citations = _citations_for(selected_chunks)

    return QueryResponse(answer=answer, citations=citations)

