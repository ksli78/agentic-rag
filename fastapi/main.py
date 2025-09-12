# -*- coding: utf-8 -*-
import io
import hashlib
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi import Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

import pdfplumber

import re
from collections import Counter,defaultdict

import lancedb
from lancedb import LanceDBConnection
from lancedb.pydantic import LanceModel, Vector


import ollama

# ---------------------------
# Environment / Config
# ---------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
GEN_MODEL   = os.getenv("OLLAMA_GEN_MODEL", "llama3.2:latest")
LANCEDB_URI = os.getenv("LANCEDB_URI", "/data/lancedb")

CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", "1200"))     # ~chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # overlap chars
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.35"))  # 0..1, raise/lower to taste

client = ollama.Client(host=OLLAMA_HOST)

app = FastAPI(title="ADAM Ingestion & Query API")

# ---------------------------
# LanceDB Schemas
# ---------------------------
class Document(LanceModel):
    doc_id: str = Field(..., description="Stable ID (SP item id or GUID)")
    source_url: Optional[str] = None
    title: Optional[str] = None
    sha256: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    page_count: Optional[int] = None
    mime_type: Optional[str] = "application/pdf"
    ingested_at: str

class Chunk(LanceModel):
    doc_id: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    embedding: Vector(768)
    # optional extras:
    # section: Optional[str] = None

# ---------------------------
# DB helpers
# ---------------------------
def get_db() -> LanceDBConnection:
    os.makedirs(LANCEDB_URI, exist_ok=True)
    return lancedb.connect(LANCEDB_URI)

def get_or_create_tables(conn: LanceDBConnection):
    # Documents table (metadata only)
    if "documents" not in conn.table_names():
        conn.create_table("documents", schema=Document, mode="overwrite")
    # Chunks table (vectors)
    if "chunks" not in conn.table_names():
        conn.create_table("chunks", schema=Chunk, mode="overwrite")
    return conn.open_table("documents"), conn.open_table("chunks")

# ---------------------------
# PDF / Text utilities
# ---------------------------
# Remove residual policy mastheads and table-like header blocks from chunks
_HDR_PATTERNS = [
    r"^policy\s*document\s*no\.:.*$",      # "Policy Document No.:"
    r"^en[-\s]?po[-\s]?\d+.*page:.*$",     # "EN-PO-0301 ... Page: 1 of 4"
    r"^accountable organization:.*$",
    r"^management system.*$",
    r"^standard operating procedure.*$",
    r"^effective date:.*$",
    r"^revision:.*$",
    r"^page:\s*\d+\s*of\s*\d+.*$",
    r"^copyright.*$",
]
_HDR_RES = [re.compile(p, re.I) for p in _HDR_PATTERNS]
# --- Section-aware splitting (simple SOP heuristics) --------------------------
_SECTION_NUM_RE = re.compile(r'^\s*(\d+(?:\.\d+){0,3})\s+([A-Z][^\n]{0,120})$')  # e.g., "6.1 Create, Modify, or Deactivate Accounts"
_EXHIBIT_RE     = re.compile(r'^\s*Exhibit\s+\d+', re.IGNORECASE)
def split_into_sections(clean_pages: List[str]) -> List[Tuple[int, int, Optional[str], str]]:
    """
    Produces section-sized blocks with accurate page spans.
    Returns a list of (page_start, page_end, section_title_or_None, section_text).
    """
    sections: List[Tuple[int,int,Optional[str],str]] = []

    def flush(curr_pages: List[Tuple[int,str]], title: Optional[str]):
        if not curr_pages:
            return
        p1 = curr_pages[0][0]
        p2 = curr_pages[-1][0]
        body = "\n\n".join([t for _, t in curr_pages]).strip()
        if body:
            sections.append((p1, p2, title, body))

    current: List[Tuple[int,str]] = []
    current_title: Optional[str] = None

    for pi, raw in enumerate(clean_pages, start=1):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # Detect a heading at the top of a page OR anywhere in lines.
        found_title = None
        for ln in lines[:6]:  # look near the top; SOP headings usually sit here
            if _SECTION_NUM_RE.match(ln) or _EXHIBIT_RE.match(ln):
                found_title = ln
                break

        if found_title:
            # start a new section
            flush(current, current_title)
            current = [(pi, raw)]
            current_title = found_title
        else:
            # continue current section
            current.append((pi, raw))

    flush(current, current_title)
    return sections

def clean_for_prompt(txt: str) -> str:
    lines = [l.strip() for l in txt.splitlines()]
    out = []
    pipe_count = 0
    for l in lines:
        if not l:
            continue
        # drop obvious header/footer lines
        if any(rx.search(l) for rx in _HDR_RES):
            continue
        # collapse “ASCII tables” (lines dominated by pipes)
        if l.count("|") >= 2:
            pipe_count += 1
            # ignore short header tables entirely (2–6 lines)
            if pipe_count <= 6:
                continue
            # otherwise keep only occasional rows
            if pipe_count % 5 != 0:
                continue
        out.append(l)
    # compact whitespace
    s = "\n".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def extract_pdf_text_and_pages(pdf_bytes: bytes) -> Tuple[str, int, List[str]]:
    """
    Return (full_text, page_count, page_texts).
    page_texts is a list where index = 0-based page number.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_texts: List[str] = []
    for page in reader.pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:
            page_texts.append("")
    full = "\n\n".join(page_texts)
    return full, len(reader.pages), page_texts


def _mkrow_md(cells):
    # render a list of cell strings into a pipe table row
    return "| " + " | ".join((cells or [])) + " |"


def _is_obvious_noise_table(headers, rows):
    """
    Heuristics to skip noisy/boilerplate tables like 'Document History' grids
    or tables that are mostly numeric counts/approval matrices.
    Tweak as needed.
    """
    hdr = " ".join(headers).lower() if headers else ""
    if any(key in hdr for key in ["document history", "change/revision", "sections affected"]):
        return True
    if "description of change" in hdr:
        return True
    # mostly numeric table?
    num_cells = sum(1 for r in rows for c in r if re.fullmatch(r"[0-9>*/\-\s.:]+", (c or "").strip()))
    total_cells = sum(len(r) for r in rows) or 1
    if num_cells / total_cells > 0.8:
        return True
    return False

def _table_to_markdown(table):
    """
    table: list of rows (lists of cells) as returned by pdfplumber.
    Returns a compact markdown string with header + separator + up to N rows.
    """
    if not table:
        return ""
    # pdfplumber returns tables as lists of rows. Use first non-empty row as header.
    rows = [[(c or "").strip() for c in row] for row in table]
    header = None
    data = []
    for r in rows:
        if header is None and any(x for x in r):
            header = r
        elif header is not None:
            data.append(r)
    if not header:
        return ""
    # Skip obvious noise tables
    if _is_obvious_noise_table(header, data):
        return ""

    # Normalize widths
    width = max(len(header), *(len(r) for r in data)) if data else len(header)
    header = header + [""] * (width - len(header))
    data = [r + [""] * (width - len(r)) for r in data]

    out = []
    out.append(_mkrow_md(header))
    out.append("| " + " | ".join(["---"] * width) + " |")
    # Cap number of rows to keep chunks light
    for r in data[:30]:
        out.append(_mkrow_md(r))
    return "\n".join(out)
def extract_pdf_text_tables(pdf_bytes: bytes) -> Tuple[str, int, List[str]]:
    """
    Use pdfplumber to extract text and (selected) tables.
    Returns (full_text, page_count, page_texts) where each page_text is
    text paragraphs followed by any useful tables converted to Markdown.
    """
    page_texts: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # page text
            t = page.extract_text() or ""

            # try extracting tables; join useful ones as markdown blocks
            md_tables = []
            try:
                tables = page.extract_tables() or []
                for tbl in tables:
                    md = _table_to_markdown(tbl)
                    if md:
                        md_tables.append(md)
            except Exception:
                pass

            if md_tables:
                t = (t.strip() + "\n\n" + "\n\n".join(md_tables)).strip()

            page_texts.append(t)

    full = "\n\n".join(page_texts)
    return full, len(page_texts), page_texts
# -------------------------------------------------------------------------------



# --- Heuristics for boilerplate removal ---------------------------------------
HEADER_FOOTER_MIN_PAGES_FRACTION = 0.6  # line must appear on >=60% pages to be considered boilerplate

# Simple normalizer: collapse spaces, unify dashes, strip extra whitespace.
_whitespace_re = re.compile(r"[ \t]+")
def _normalize_line(s: str) -> str:
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = _whitespace_re.sub(" ", s).strip()
    return s

# Obvious noise patterns (tweak over time)
_NOISE_PATTERNS = [
    r"^page\s*\d+\s*(of\s*\d+)?$",
    r"^en[- ]?po[- ]?\d+.*(page).*of.*$",    # doc/page bars like "EN-PO-0301 Page: 1 of 4"
    r"^this document contains proprietary information.*$",  # legal footer
    r"^uncontrolled if printed.*$",
    r"^copyright\s*©\s*\d{4}.*$",            # copyright bars
    r"^management system.*$",
    r"^policy$|^standard operating procedure$",
]

_NOISE_RES = [re.compile(pat, re.I) for pat in _NOISE_PATTERNS]

def _is_noise_line(line: str) -> bool:
    norm = _normalize_line(line)
    if not norm:
        return True
    for rx in _NOISE_RES:
        if rx.search(norm):
            return True
    # ultra-short lines with only punctuation
    if len(norm) <= 2 or re.fullmatch(r"[-–—_]+", norm):
        return True
    return False
def _is_long_line(s: str) -> bool:
    return len(s) >= 30 and len(s.split()) >= 4

def remove_boilerplate(page_texts: List[str]) -> List[str]:
    """
    Remove header/footer lines that repeat across most pages, and obvious noise lines.
    Return cleaned page_texts (one string per page).
    """
    # 1) Split pages into lines + normalize
    pages_lines = [[_normalize_line(l) for l in p.splitlines()] for p in page_texts]

    # 2) Count how often each line appears across pages (set per page to avoid double-counting)
    line_occurs = Counter()
    for lines in pages_lines:
        uniq = set(l for l in lines if l)
        line_occurs.update(uniq)

    min_pages = max(1, int(len(page_texts) * HEADER_FOOTER_MIN_PAGES_FRACTION))
    boilerplate = {
        line for line, cnt in line_occurs.items()
        if cnt >= min_pages and _is_long_line(line)  # <-- add this guard
    }

    # 3) Rebuild each page without boilerplate + without obvious noise lines
    cleaned_pages = []
    for i, lines in enumerate(pages_lines):
        kept = []
        for l in lines:
            if not l:
                continue
            if l in boilerplate:
                continue
            if _is_noise_line(l):
                continue
            kept.append(l)

        # --- Fallback: if kept content is too small, keep the original page text
        original = "\n".join(lines).strip()
        cleaned   = "\n".join(kept).strip()

        # heuristic: if cleaned is < 30% of original length, or fewer than 3 non-empty lines,
        # fall back to original (prevents total wipe-outs on real content pages)
        if len(cleaned) < 0.3 * max(1, len(original)) or sum(1 for x in kept if x) < 3:
            cleaned_pages.append(original)
        else:
            cleaned_pages.append(cleaned)
    return cleaned_pages

def build_page_offsets(page_texts: List[str]) -> List[int]:
    """
    Build cumulative character offsets for each page start, based on the
    `full = "\n\n".join(page_texts)` concatenation used above.
    Returns a list of length (pages + 1):
      offsets[p] = char index where page p (0-based) starts in `full`
      offsets[-1] = len(full)
    """
    offsets: List[int] = [0]
    running = 0
    for i, t in enumerate(page_texts):
        # Each page in `full` is joined by "\n\n", except before the first page.
        if i > 0:
            running += 2  # for the "\n\n" we inserted
        offsets.append(running + len(t))
        running = offsets[-1]
    # Convert to true "start" offsets (prefix sums shifted by one)
    # After the loop, offsets[i] holds the end-of-page-i-1; we need starts.
    # Recompute into starts in-place:
    starts: List[int] = [0]
    total = 0
    for i, t in enumerate(page_texts):
        if i > 0:
            total += 2  # the "\n\n"
        starts.append(total)
        total += len(t)
    # `starts` length is pages+1; starts[p] is start char of page p; starts[-1] = len(full)
    return starts

def char_span_to_pages(start_char: int, end_char: int, page_offsets: List[int]) -> Tuple[int, int]:
    """
    Map a [start_char, end_char) span in `full` to 1-based (page_start, page_end),
    using `page_offsets` from build_page_offsets (0-based page starts).
    """
    # Find page index such that offsets[i] <= pos < offsets[i+1]
    def find_page(pos: int) -> int:
        # linear scan is fine for modest page counts; optimize to binary search if needed
        for i in range(len(page_offsets) - 1):
            if page_offsets[i] <= pos < page_offsets[i + 1]:
                return i
        return len(page_offsets) - 2  # clamp to last page

    p_start = find_page(start_char) + 1  # to 1-based
    p_end = find_page(max(end_char - 1, start_char)) + 1
    return p_start, p_end

def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    page_offsets: Optional[List[int]] = None
) -> List[Tuple[int, int, str]]:
    """
    Return a list of (page_start, page_end, chunk_text).

    - If `page_offsets` is provided (from build_page_offsets), we compute actual
      page ranges by mapping the chunk's character span [start, end) into pages.
    - If not provided, we fall back to 1..1 so ingest_text still works.
    """
    start = 0
    chunks: List[Tuple[int, int, str]] = []
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        if page_offsets:
            p1, p2 = char_span_to_pages(start, end, page_offsets)
        else:
            p1 = p2 = 1
        chunks.append((p1, p2, chunk))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# ---------------------------
# Embeddings / Generation
# ---------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for t in texts:
        # Newer ollama client uses "input" (plural) and returns "embeddings"
        # but some versions expect "prompt". Try both.
        try:
            out = client.embeddings(model=EMBED_MODEL, input=t)
            vec = out["embeddings"][0] if "embeddings" in out else out["embedding"]
        except Exception:
            out = client.embeddings(model=EMBED_MODEL, prompt=t)  # fallback
            vec = out.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError("Embedding vector missing or wrong type")
        vectors.append(vec)
    return vectors

def answer_with_context(question: str, contexts: List[Dict[str, Any]]) -> str:
    # Build a grounded prompt
    parts = ["You are ADAM, a helpful assistant. Use the CONTEXT to answer."]
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"[{i}] doc_id={ctx['doc_id']} pgs {ctx['page_start']}-{ctx['page_end']}:\n{clean_for_prompt(ctx['text'])[:1200]}")
    parts.append(f"Question: {question}\nAnswer clearly with citations like [1], [2] where used.")
    prompt = "\n\n".join(parts)

    resp = client.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return resp["message"]["content"]

# ---------------------------
# Pydantic payloads
# ---------------------------
class IngestUrl(BaseModel):
    url: str
    doc_id: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    title: Optional[str] = None

class IngestText(BaseModel):
    doc_id: str
    text: str
    source_url: Optional[str] = None
    title: Optional[str] = None

class QueryPayload(BaseModel):
    question: str
    top_k: int = 6
    debug: bool = False  # new optional flag; default is False

class DocumentItem(BaseModel):
    doc_id: str
    source_url: Optional[str] = None
    title: Optional[str] = None
    sha256: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    page_count: Optional[int] = None
    mime_type: Optional[str] = None
    ingested_at: str

class ChunkItem(BaseModel):
    doc_id: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str

class ClearDBPayload(BaseModel):
    confirm:str
class DeleteDocPayload(BaseModel):
    confirm:str
    doc_id:str
# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    try:
        # lightweight ping to ollama
        client.list()  # raises if unreachable
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
@app.post("/ingest/url")
async def ingest_url(payload: IngestUrl):
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as s:
        r = await s.get(payload.url)
        r.raise_for_status()
        pdf_bytes = r.content

    return _ingest_common(
        pdf_bytes=pdf_bytes,
        doc_id=payload.doc_id,
        source_url=payload.url,
        etag=payload.etag,
        last_modified=payload.last_modified,
        title=payload.title
    )
@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    etag: Optional[str] = Form(None),
    last_modified: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
):
    pdf_bytes = await file.read()
    return _ingest_common(
        pdf_bytes=pdf_bytes,
        doc_id=doc_id,
        source_url=source_url,
        etag=etag,
        last_modified=last_modified,
        title=title or file.filename
    )
@app.post("/ingest/text")
def ingest_text(payload: IngestText):
    text = payload.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    doc_id = payload.doc_id or hashlib.sha1(text.encode("utf-8")).hexdigest()
    sha = sha256_bytes(text.encode("utf-8"))
    page_count = 1

    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)

    # short-circuit if unchanged
    existing = list(docs.search().where(f"doc_id == '{doc_id}'").to_list())
    if existing and existing[0]["sha256"] == sha:
        return {"status": "skipped", "reason": "unchanged", "doc_id": doc_id}

    # delete any previous chunks for this doc_id
    chunks_tbl.delete(f"doc_id == '{doc_id}'")

    # chunk and embed
    ch = chunk_text(text)
    texts = [c[2] for c in ch]
    vecs = embed_texts(texts)

    # insert chunks
    rows = []
    for (pgs, pge, t), v in zip(ch, vecs):
        rows.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + t[:64]).encode("utf-8")).hexdigest(),
            "page_start": pgs,
            "page_end": pge,
            "text": t,
            "embedding": v
        })
    chunks_tbl.add(rows)

    # upsert doc metadata
    docs.delete(f"doc_id == '{doc_id}'")
    docs.add([{
        "doc_id": doc_id,
        "source_url": payload.source_url,
        "title": payload.title,
        "sha256": sha,
        "etag": None,
        "last_modified": None,
        "page_count": page_count,
        "ingested_at": datetime.utcnow().isoformat() + "Z"
    }])

    return {"status": "ok", "doc_id": doc_id, "chunks": len(rows)}
@app.post("/query")
def query(payload: QueryPayload):
    # Open both tables: documents (metadata) and chunks (vectors)
    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)

    # Embed the question
    qvec = embed_texts([payload.question])[0]

    # Vector search (same top_k as before)
    # hits = chunks_tbl.search(qvec).limit(payload.top_k).to_list()
    # Fetch a larger candidate pool for hybrid ranking
    # We use max(top_k * 5, 50) to ensure enough docs for RRF.
    vector_limit = max(payload.top_k * 5, 50)
    hits_vector = chunks_tbl.search(qvec).limit(vector_limit).to_list()
    
    if not hits_vector:
        return {"answer": "I couldn't find relevant content.", "citations": []}
    
    def _score(h: dict) -> Optional[float]:
        # Some LanceDB versions return "score" (higher=better), others only "_distance" (lower=better).
        if "score" in h and h["score"] is not None:
            return float(h["score"])
        if "_distance" in h and h["_distance"] is not None:
            return 1.0 / (1.0 + float(h["_distance"]))
        return None
    def lexical_score(text: str, query: str) -> int:
        """
        Score by presence of expanded key phrases/words (case-insensitive).
        Each matched phrase/word contributes 1.
        """
        t = text.lower()
        score = 0
        for term in expand_lex_terms(query):
            if term in t:
                score += 1
        return score
        # --- simple synonym expansion for HR terms -------------------------
    
    LEX_SYNONYMS = {
        "pto": ["pto", "paid time off", "time off", "leave"],
        "workweek": ["workweek", "work week"],
        # add more as needed
    }
    def expand_lex_terms(q: str) -> list[str]:
        base = q.lower()
        terms = set()
        for k, alts in LEX_SYNONYMS.items():
            if k in base:
                terms.update(alts)
        # always include the original words as fallbacks
        terms.update(base.split())
        # drop 1-char tokens
        return [t for t in terms if len(t) > 1]
  
  
 

    # --- Hybrid ranking with reciprocal-rank fusion -------------------------------
    # 1) Build a rank map for vector hits (rank 1 = best)
    vector_rank = {h['chunk_id']: i + 1 for i, h in enumerate(hits_vector)}

    # 2) Compute lexical scores for each candidate
    lex_scores = {h['chunk_id']: lexical_score(h['text'], payload.question) for h in hits_vector}

    # 3) Rank candidates by lexical score (highest score gets rank 1)
    #    Ties are implicitly broken by original vector ordering (stable sort).
    lex_sorted = sorted(hits_vector, key=lambda h: lex_scores[h['chunk_id']], reverse=True)
    lex_rank = {h['chunk_id']: i + 1 for i, h in enumerate(lex_sorted)}

    # 4) Compute reciprocal-rank fusion (RRF) score: 1/(vector_rank) + 1/(lex_rank)
    rrf_scores = {}
    for h in hits_vector:
        cid = h['chunk_id']
        rv = vector_rank[cid]
        rl = lex_rank[cid]
        rrf_scores[cid] = 1.0 / rv + 1.0 / rl

    # 5) Sort by RRF score (descending) and keep only top_k
    #hits = sorted(hits_vector, key=lambda h: rrf_scores[h['chunk_id']], reverse=True)[:payload.top_k]
    # Sort chunks by fused score (desc)
    ranked = sorted(hits_vector, key=lambda h: rrf_scores[h['chunk_id']], reverse=True)
    # --- Doc-level gating: choose the best document(s) first ----------------------
   
    # --------------------------from collections import defaultdict

    doc_stats = defaultdict(lambda: {"lex_sum": 0, "max_rrf": 0.0})
    for h in hits_vector:
        did = h["doc_id"]
        cid = h["chunk_id"]
        doc_stats[did]["lex_sum"] += max(0, lex_scores.get(cid, 0))
        doc_stats[did]["max_rrf"] = max(doc_stats[did]["max_rrf"], rrf_scores[cid])

    # Pick the single best doc by (has lexical hits first, then fused strength)
    def _doc_key(item):
        did, s = item
        return (s["lex_sum"] > 0, s["max_rrf"])

    best_doc = None
    if doc_stats:
        best_doc = max(doc_stats.items(), key=_doc_key)[0]

    # Keep ONLY the best doc if it has any lexical signal; otherwise keep top 2 docs by fused
    if best_doc and doc_stats[best_doc]["lex_sum"] > 0:
        allowed_docs = {best_doc}
    else:
        # no lexical signal → allow top 2 by fused as fallback
        top2 = sorted(doc_stats.items(), key=_doc_key, reverse=True)[:2]
        allowed_docs = {d for d, _ in top2}

    # Filter ranked chunks to allowed docs
    ranked = [h for h in ranked if h["doc_id"] in allowed_docs]

    # Keep best N per doc (as before)
    BEST_PER_DOC = 2
    per_doc = defaultdict(list)
    for h in ranked:
        if len(per_doc[h["doc_id"]]) < BEST_PER_DOC:
            per_doc[h["doc_id"]].append(h)

    # Flatten in doc order (best doc first)
    hits = []
    for did, _ in sorted(((d, doc_stats[d]) for d in allowed_docs), key=_doc_key, reverse=True):
        hits.extend(per_doc.get(did, []))
    hits = hits[:payload.top_k]
    

    # Only now that lex_scores and hits exist, decide on cutoff
    has_lex_hits = any(v > 0 for v in lex_scores.values())
    if not has_lex_hits:
        scores = [s for s in (_score(h) for h in hits) if s is not None]
        best = max(scores) if scores else 0.0
        if best < MIN_SIMILARITY:
            return {
                "answer": "I couldn't find a strong match for your question.",
                "citations": [],
                **({"debug": {"vector_hits": len(hits_vector), "best_fused_score": best}} if getattr(payload, "debug", False) else {})
            }
    #-----------------------------------------------------
    
    doc_meta = {}
    for h in hits:
        did = h.get("doc_id")
        if not did or did in doc_meta:
            continue
        rows = docs_tbl.search().where(f"doc_id == '{did}'").to_list()
        if rows:
            r = rows[0]
            doc_meta[did] = {
                "title": r.get("title"),
                "source_url": r.get("source_url"),
            }
        else:
            doc_meta[did] = {"title": None, "source_url": None}

    # Build contexts for the LLM and richer citations for the response
    contexts = []
    citations = []
    for h in hits:
        did = h["doc_id"]
        p1 = int(h.get("page_start", 1))
        p2 = int(h.get("page_end", 1))
        contexts.append({
            "doc_id": did,
            "page_start": p1,
            "page_end": p2,
            "text": h["text"],
        })
        citations.append({
            "doc_id": did,
            "title": doc_meta[did]["title"],
            "source_url": doc_meta[did]["source_url"],
            "page_start": p1,
            "page_end": p2,
        })
    # Only cite from allowed docs (safety net)
    allowed_doc_ids = {h["doc_id"] for h in hits}
    citations = [c for c in citations if c["doc_id"] in allowed_doc_ids]
    
    # De-duplicate identical doc/page citations
    seen = set()
    dedup = []
    for c in citations:
        key = (c["doc_id"], c["page_start"], c["page_end"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    citations = dedup[:5]  # cap to 5 neat cites
    
    # Generate the answer as before
    answer = answer_with_context(payload.question, contexts)

    # Return answer with human-friendly citations
    resp = {"answer": answer, "citations": citations}
    # If caller asked for debug info, include stats about the search
    if payload.debug:
        resp["debug"] = {
            "vector_hits": len(hits_vector),                        # how many chunks were scanned
            "top_scores": [_score(h) for h in hits]          # scores for the fetched chunks
        }
    return resp
@app.get("/documents")
def list_documents(limit: int = 100):
    """
    List up to `limit` documents with their stored metadata.
    """
    conn = get_db()
    docs_tbl, _ = get_or_create_tables(conn)

    # Simple capped list. (If you need paging later, we can add it.)
    rows = docs_tbl.search().limit(max(1, min(limit, 1000))).to_list()
    # Normalize to our response model shape
    items = []
    for r in rows:
        items.append(DocumentItem(
            doc_id=r.get("doc_id"),
            source_url=r.get("source_url"),
            title=r.get("title"),
            sha256=r.get("sha256"),
            etag=r.get("etag"),
            last_modified=r.get("last_modified"),
            page_count=r.get("page_count"),
            mime_type=r.get("mime_type"),
            ingested_at=r.get("ingested_at"),
        ).dict())
    return {"count": len(items), "items": items}
@app.get("/documents/{doc_id}")
def get_document_with_chunks(doc_id: str):
    """
    Return one document's metadata and all of its chunks.
    """
    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)

    doc_rows = docs_tbl.search().where(f"doc_id == '{doc_id}'").to_list()
    if not doc_rows:
        raise HTTPException(status_code=404, detail="Document not found")

    # Fetch all chunks for this document (cap at a large number to avoid surprises)
    ch_rows = chunks_tbl.search().where(f"doc_id == '{doc_id}'").limit(100000).to_list()

    # Format outputs
    d = doc_rows[0]
    document = DocumentItem(
        doc_id=d.get("doc_id"),
        source_url=d.get("source_url"),
        title=d.get("title"),
        sha256=d.get("sha256"),
        etag=d.get("etag"),
        last_modified=d.get("last_modified"),
        page_count=d.get("page_count"),
        mime_type=d.get("mime_type"),
        ingested_at=d.get("ingested_at"),
    ).dict()

    chunks = []
    for c in ch_rows:
        chunks.append(ChunkItem(
            doc_id=c.get("doc_id"),
            chunk_id=c.get("chunk_id"),
            page_start=int(c.get("page_start", 1)),
            page_end=int(c.get("page_end", 1)),
            text=c.get("text", ""),
        ).dict())

    return {
        "document": document,
        "chunk_count": len(chunks),
        "chunks": chunks
    }
@app.delete("/documents/{doc_id}")
def delete_document(payload:DeleteDocPayload):
    """
    Delete ONE document and all its chunks.
    Safety check: requires confirm=DELETE (query param) to proceed.
    Example: DELETE /documents/22da2d3c-...-f714300e3ddf?confirm=DELETE
    """
    if payload.confirm != "DELETE":
        raise HTTPException(status_code=400, detail="Confirmation missing. Append ?confirm=DELETE to proceed.")

    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)

    # Count chunks before delete (best-effort; ignore large-corpus cost)
    to_del = list(chunks_tbl.search().where(f"doc_id == '{payload.doc_id}'").limit(1_000_000).to_list())
    chunk_count = len(to_del)

    # Delete chunks then the document metadata
    chunks_tbl.delete(f"doc_id == '{payload.doc_id}'")
    docs_tbl.delete(f"doc_id == '{payload.doc_id}'")

    return {"status": "ok", "doc_id": payload.doc_id, "deleted_chunks": chunk_count}

from fastapi import Query

class ReindexPayload(BaseModel):
    mode: str = "full"    # "full" (re-download & re-extract) or "reembed"
@app.post("/reindex/{doc_id}")
async def reindex_document(doc_id: str, payload: ReindexPayload):
    """
    Re-run extraction/chunking/embedding for ONE document.

    modes:
      - "full" (default): re-download from source_url and fully re-extract
      - "reembed": if no source_url (or download fails), rebuild from existing chunks:
                   concatenate text -> clean -> re-chunk -> re-embed
    """
    conn = get_db()
    docs_tbl, chunks_tbl = get_or_create_tables(conn)

    # Look up doc row
    rows = docs_tbl.search().where(f"doc_id == '{doc_id}'").to_list()
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")
    d = rows[0]
    source_url = d.get("source_url")
    title      = d.get("title")
    etag       = d.get("etag")
    last_mod   = d.get("last_modified")

    mode = (payload.mode or "full").lower().strip()

    # Preferred path: FULL re-extract (if we have a source_url)
    if mode in ("full", "auto") and source_url:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as s:
                r = await s.get(source_url)
                r.raise_for_status()
                pdf_bytes = r.content
        except Exception as ex:
            if mode == "full":
                raise HTTPException(status_code=502, detail=f"Download failed from source_url: {ex}")
            pdf_bytes = None
        if pdf_bytes:
            # Re-ingest using the same doc_id + metadata, forcing past the SHA check
            return _ingest_common(
                pdf_bytes=pdf_bytes,
                doc_id=doc_id,
                source_url=source_url,
                etag=etag,
                last_modified=last_mod,
                title=title,
                force=True
            )

    # Fallback: RE-EMBED from existing chunk text (no PDF needed)
    #  - Pull current chunks, rebuild text, re-clean+re-chunk, re-embed.
    ch_rows = chunks_tbl.search().where(f"doc_id == '{doc_id}'").limit(1_000_000).to_list()
    if not ch_rows:
        raise HTTPException(status_code=400, detail="No chunks to re-embed and no source_url available.")

    # Rebuild a "document" string in page order if possible
    # Group by (page_start), then concatenate
    ch_rows_sorted = sorted(ch_rows, key=lambda r: (int(r.get("page_start", 1)), r.get("chunk_id", "")))
    rebuilt_pages: Dict[int, List[str]] = {}
    for r in ch_rows_sorted:
        p = int(r.get("page_start", 1))
        rebuilt_pages.setdefault(p, []).append(r.get("text", ""))

    page_texts = ["\n".join(rebuilt_pages[p]) for p in sorted(rebuilt_pages.keys())]
    # Clean using your existing boilerplate remover
    clean_pages = remove_boilerplate(page_texts)
    text = "\n\n".join(clean_pages)
    page_offsets = build_page_offsets(clean_pages)
    new_chunks = chunk_text(text, page_offsets=page_offsets)

    # Re-embed & replace chunks
    texts = [c[2] for c in new_chunks]
    vecs = embed_texts(texts)

    # Delete existing chunks for this doc
    chunks_tbl.delete(f"doc_id == '{doc_id}'")

    rows_to_add = []
    for (pgs, pge, t), v in zip(new_chunks, vecs):
        header = make_chunk_header(doc_id, title, pgs, pge)
        rows_to_add.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + t[:64]).encode("utf-8")).hexdigest(),
            "page_start": pgs,
            "page_end": pge,
            "text": f"{header}\n\n{t}",
            "embedding": v
        })
    chunks_tbl.add(rows_to_add)

    # Keep document row as-is; update timestamp so you can see it changed
    docs_tbl.delete(f"doc_id == '{doc_id}'")
    docs_tbl.add([{
        "doc_id": doc_id,
        "source_url": source_url,
        "title": title,
        "sha256": d.get("sha256"),        # unchanged here (we didn't re-hash the original bytes)
        "etag": etag,
        "last_modified": last_mod,
        "page_count": d.get("page_count"),
        "mime_type": d.get("mime_type"),
        "ingested_at": datetime.utcnow().isoformat() + "Z"
    }])

    return {"status": "ok", "doc_id": doc_id, "mode": "reembed", "chunks": len(rows_to_add)}
@app.post("/admin/clear_db")
def clear_db(payload:ClearDBPayload):
    """
    DANGER: Clears ALL data in LanceDB (documents + chunks).
    Call with: { "confirm": "ERASE" }
    """
    if not isinstance(payload, ClearDBPayload) or payload.confirm != "ERASE":
        raise HTTPException(status_code=400, detail="Confirmation missing. Send {\"confirm\":\"ERASE\"} to proceed.")

    conn = get_db()

    # Recreate tables in-place to guarantee a clean slate
    # (mode='overwrite' will drop existing data and recreate the table)
    conn.create_table("documents", schema=Document, mode="overwrite")
    conn.create_table("chunks", schema=Chunk, mode="overwrite")

    return {"status": "ok", "message": "All tables cleared (documents, chunks)."}

def make_chunk_header(doc_id: str, title: Optional[str], p1: int, p2: int, section: Optional[str] = None) -> str:
    """
    Build a short, authoritative header that improves retrieval and reranking.
    Example: "Access Control | 386b...e14 | p.2-3"  (title | doc_id | page range)
    """
    parts = []
    if title: parts.append(title.strip())
    if section: parts.append(section.strip())
    parts.append(doc_id)
    parts.append(f"p.{p1}-{p2}")
    return " | ".join(parts)


# ---------------------------
# Internal common ingest
# ---------------------------
def _ingest_common(
    pdf_bytes: bytes,
    doc_id: Optional[str],
    source_url: Optional[str],
    etag: Optional[str],
    last_modified: Optional[str],
    title: Optional[str],
    force:bool =False
):
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    
    text, page_count, page_texts = extract_pdf_text_tables(pdf_bytes)
    sha = sha256_bytes(pdf_bytes)
   
    
    doc_id = doc_id or hashlib.sha1((source_url or sha)[:64].encode("utf-8")).hexdigest()

    conn = get_db()
    docs, chunks_tbl = get_or_create_tables(conn)

    # unchanged short-circuit
    existing = list(docs.search().where(f"doc_id == '{doc_id}'").to_list())
    if  (not force) and existing and existing[0]["sha256"] == sha:
        return {"status": "skipped", "reason": "unchanged", "doc_id": doc_id}

    # delete any previous chunks for this doc_id
    chunks_tbl.delete(f"doc_id == '{doc_id}'")
    
    # --- Clean: remove repeated headers/footers + obvious noise
    clean_pages = remove_boilerplate(page_texts)
    
    # Rebuild full text from cleaned pages (preserving page joins with \n\n)
    #text = "\n\n".join(clean_pages)
    
    # Build offsets so we can map chunk character positions to real pages
    #page_offsets = build_page_offsets(clean_pages)
    # chunk + embed
    #ch = chunk_text(text, page_offsets=page_offsets)
    #texts = [c[2] for c in ch]

    # 1) Split into SOP sections (parent chunks)
    parent_sections = split_into_sections(clean_pages)

    # 2) For each section, only sub-split if it greatly exceeds size
    ch: List[Tuple[int,int,str,str]] = []  # (p1,p2,section_title,chunk_text)
    for (p1, p2, sec_title, sec_text) in parent_sections:
        if len(sec_text) <= CHUNK_SIZE * 2:
            ch.append((p1, p2, sec_title or None, sec_text))
        else:
            # sub-split within this section (preserve its page span using local offsets)
            local_pages = [sec_text]  # treat as one "page" for mapping
            local_offsets = build_page_offsets(local_pages)
            for (sp1, sp2, sub) in chunk_text(sec_text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, page_offsets=local_offsets):
                # keep the section page range, not the local 1..1
                ch.append((p1, p2, sec_title or None, sub))

    texts = [c[3] for c in ch]
    vecs = embed_texts(texts)

    rows = []
    for (pgs, pge, sec_title, t), v in zip(ch, vecs):
        header = make_chunk_header(doc_id, title, pgs, pge, section=sec_title)
        rows.append({
            "doc_id": doc_id,
            "chunk_id": hashlib.sha1((doc_id + t[:64]).encode("utf-8")).hexdigest(),
            "page_start": pgs,
            "page_end": pge,
            "text": f"{header}\n\n{t}",
            "embedding": v
        })
    chunks_tbl.add(rows)

    # upsert document metadata
    docs.delete(f"doc_id == '{doc_id}'")
    docs.add([{
        "doc_id": doc_id,
        "source_url": source_url,
        "title": title,
        "sha256": sha,
        "etag": etag,
        "last_modified": last_modified,
        "page_count": page_count,
        "ingested_at": datetime.utcnow().isoformat() + "Z"
    }])

    return {"status": "ok", "doc_id": doc_id, "chunks": len(rows)}
