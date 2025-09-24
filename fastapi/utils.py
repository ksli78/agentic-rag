# fastapi/utils.py
import hashlib
import json
import httpx
import logging
import numpy as np
from typing import List
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
        if "data" in data and isinstance(data["data"], list) and data["data"] and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
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
    paras = md.split("\n\n")
    chunks: List[str] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) <= block_size:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = min(start + block_size, len(p))
                nl = p.rfind("\n", start, end)
                cut = nl if nl > start + 100 else end
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
            {"role": "user", "content": user_prompt}
        ],
        "options": {"temperature": float(temperature or 0)}
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        text = r.content.decode()  # get the raw body

    result = []
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