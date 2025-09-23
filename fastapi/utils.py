import hashlib
import httpx
import numpy as np
from typing import List, Tuple

from config import settings


# ---------- LLM & Embeddings (Ollama) ----------

async def _embed_one(text: str) -> List[float]:
    url = f"{settings.ollama_host}/api/embeddings"
    payload = {"model": settings.ollama_embed_model, "input": text}
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    # shape handling
    if isinstance(data, dict):
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        if "data" in data and isinstance(data["data"], list) and data["data"] and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
            return data["embeddings"][0]
    raise ValueError(f"Unexpected embedding response: {data}")

async def embed_texts(texts: List[str]) -> List[List[float]]:
    # Some Ollama versions don’t handle list inputs reliably—embed sequentially.
    out: List[List[float]] = []
    for t in texts:
        out.append(await _embed_one(t))
    return out

async def chat_complete(system_prompt: str, user_prompt: str, temperature: float) -> str:
    url = f"{settings.ollama_host}/api/chat"
    payload = {
        "model": settings.ollama_gen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "options": {"temperature": temperature}
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    # Ollama streams chunks or returns full; handle both
    if isinstance(data, dict) and "message" in data:
        return data["message"]["content"]
    if isinstance(data, list):
        return "".join([chunk.get("message", {}).get("content", "") for chunk in data])
    return str(data)

# ---------- Helpers ----------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def split_markdown_blocks(md: str, block_size: int) -> List[str]:
    """Split markdown on paragraph boundaries, further slicing long blocks."""
    paras = md.split("\n\n")
    chunks: List[str] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) <= block_size:
            chunks.append(p)
        else:
            # slice but keep line boundaries when possible
            start = 0
            while start < len(p):
                end = min(start + block_size, len(p))
                # try not to cut a line mid-way
                nl = p.rfind("\n", start, end)
                cut = nl if nl > start + 100 else end
                chunks.append(p[start:cut].strip())
                start = cut
    return chunks
