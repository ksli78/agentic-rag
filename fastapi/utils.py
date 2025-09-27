"""Utility functions for embeddings, hashing and LLM interaction.

This module centralises calls to the local Ollama server for
embedding generation and text generation.  It also contains helper
functions for computing SHA256 hashes of bytes, splitting Markdown
into approximately equal sized blocks and computing cosine
similarities.  These utilities wrap ``httpx`` calls so that the rest
of the application does not need to know about the network details.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import List

import httpx
import numpy as np

# Import settings from the top-level config module to avoid relative import issues
from config import settings


log = logging.getLogger("api.utils")


# ---------- LLM & Embeddings (Ollama) ----------

async def _embed_one_ollama(text: str) -> List[float]:
    url = f"{settings.ollama_host}/api/embeddings"
    # Use "prompt" for max compatibility across Ollama builds
    payload = {"model": settings.ollama_embed_model, "prompt": text}
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    if isinstance(data, dict):
        log.debug(f"Ollama embeddings response keys: {list(data.keys())}")
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        if (
            "data" in data
            and isinstance(data["data"], list)
            and data["data"]
            and "embedding" in data["data"][0]
        ):
            return data["data"][0]["embedding"]
        if (
            "embeddings" in data
            and isinstance(data["embeddings"], list)
            and data["embeddings"]
        ):
            return data["embeddings"][0]

    raise ValueError(f"Unexpected Ollama embeddings response: {data}")


async def embed_texts(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    for t in texts:
        vec = await _embed_one_ollama(t)
        if not vec:
            raise RuntimeError("Ollama returned an empty embedding vector.")
        out.append(vec)
    return out


# ---------- Helpers ----------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def split_markdown_blocks(md: str, block_size: int) -> List[str]:
    """
    Split Markdown into logical blocks that preserve structure:
      - A heading and its immediate list items stay together
      - A short intro paragraph immediately following a heading is attached
      - Long blocks are split at newline boundaries up to `block_size`
    """
    import re

    lines = md.splitlines()
    paras: List[str] = []
    current: List[str] = []
    heading_accum: List[str] = []  # holds a heading and its list items (+ optional short intro)

    # tweakable heuristics
    MAX_INTRO_LINES = 2  # how many non-list lines after a heading to keep with it

    def flush_heading():
        """Append accumulated heading + list + (optional short intro) as one paragraph."""
        nonlocal heading_accum
        if heading_accum:
            paras.append("\n".join(heading_accum).strip())
            heading_accum = []

    def flush_current():
        """Append accumulated normal lines as one paragraph."""
        nonlocal current
        if current:
            paras.append("\n".join(current).strip())
            current = []

    # pass 1: build structural paragraphs
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        is_heading = stripped.startswith("#")
        is_list = (
            stripped.startswith("-") or
            stripped.startswith("*") or
            re.match(r"^\d+[.)]\s+", stripped) is not None
        )

        if is_heading:
            # finish any previous paragraph
            flush_heading()
            flush_current()

            # start a new heading block
            heading_accum = [line]

            # attach immediate list items
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if nxt == "":
                    break
                if (
                    nxt.startswith("-") or nxt.startswith("*") or
                    re.match(r"^\d+[.)]\s+", nxt) is not None
                ):
                    heading_accum.append(lines[j])
                    j += 1
                else:
                    break

            # optionally attach a short “intro” paragraph if it immediately follows the heading and no list items were found
            if j == i + 1:  # no list items; look ahead for 1–2 short lines
                intro_count = 0
                k = j
                while k < len(lines) and intro_count < MAX_INTRO_LINES:
                    nxt = lines[k].strip()
                    if nxt == "":
                        break
                    # stop if next is a new heading or list
                    if nxt.startswith("#") or nxt.startswith("-") or nxt.startswith("*") or re.match(r"^\d+[.)]\s+", nxt):
                        break
                    heading_accum.append(lines[k])
                    intro_count += 1
                    k += 1
                j = k

            # advance
            i = j
            continue

        elif heading_accum and is_list:
            # list following a heading—keep accumulating
            heading_accum.append(line)
            i += 1
            continue

        elif stripped == "":
            # blank line ends whichever block is open
            flush_heading()
            flush_current()
            i += 1
            continue

        else:
            # normal text line (not a heading, not a list)
            if heading_accum:
                # If we ended up here we already attached an intro; start a new block
                flush_heading()
            current.append(line)
            i += 1

    # flush remaining
    flush_heading()
    flush_current()

    # pass 2: split any too-long paragraphs by block_size, cutting at newlines where possible
    chunks: List[str] = []
    for p in paras:
        if len(p) <= block_size:
            chunks.append(p)
            continue

        start = 0
        L = len(p)
        while start < L:
            end = min(start + block_size, L)
            # prefer to cut at a newline boundary if one exists in the window
            nl = p.rfind("\n", start, end)
            cut = nl if (nl is not None and nl > start + 100) else end
            chunks.append(p[start:cut].strip())
            start = cut

    return chunks



async def chat_complete(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call the Ollama chat API and return the response text."""
    url = f"{settings.ollama_host}/api/chat"
    payload = {
        "model": settings.ollama_gen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": float(temperature or 0)},
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        text = r.content.decode()  # get the raw body

    result: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # skip any malformed line
            continue
        # Ollama sometimes returns the answer in obj["message"]["content"],
        # or in obj["response"] (depending on version).
        if isinstance(obj, dict):
            if "message" in obj and obj["message"]:
                result.append(obj["message"].get("content", ""))
            elif "response" in obj:
                result.append(obj["response"])
    return "".join(result)


from typing import Optional, List
import json

async def extract_category_keywords(document_text: str) -> tuple[Optional[str], Optional[List[str]]]:
    """
    Extract a single category and a list of keywords from the provided document text.

    This helper uses the local Ollama chat model via ``chat_complete`` to infer
    the document's category (e.g. ``policy``, ``procedure``) and up to five
    concise keywords that describe its content.  The function truncates the
    input text to the first 4096 characters to avoid hitting context limits.
    It expects the model to return a JSON string with two keys: ``category``
    and ``keywords``.  If parsing fails, ``None`` values are returned.
    """
    excerpt = document_text[:4096] if document_text else ""
    if not excerpt:
        return None, None

    system_prompt = (
        "You are a classification assistant for internal company documents.\n"
        "Given a document excerpt, identify one broad category (such as "
        "'policy', 'procedure', 'form', or 'other') and up to five concise "
        "keywords that capture its main topics. Respond strictly in JSON with "
        "two keys: 'category' and 'keywords'.\n"
        "Example response: {\"category\": \"policy\", "
        "\"keywords\": [\"paid time off\", \"leave\", \"employees\"]}"
    )
    user_prompt = f"Document excerpt:\n{excerpt}\n\nProvide the category and keywords."

    try:
        # call the existing chat_complete helper; temperature 0 to keep deterministic
        response = await chat_complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
    except Exception as ex:
        log.error(f"Metadata extraction failed: {ex}")
        return None, None

    try:
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        log.warning(f"Failed to parse JSON from metadata extraction: {response}")
        return None, None

    category: Optional[str] = None
    keywords_list: Optional[List[str]] = None

    if isinstance(data, dict):
        cat = data.get("category")
        if isinstance(cat, str) and cat.strip():
            category = cat.strip().lower()
        kw = data.get("keywords")
        if isinstance(kw, list) and kw:
            cleaned = [str(k).strip().lower() for k in kw if str(k).strip()]
            keywords_list = cleaned if cleaned else None

    return category, keywords_list
