"""Persistent hybrid index for RAG retrieval.

This module defines the ``IndexStore`` class which encapsulates a
lightweight hybrid retrieval index.  Dense vectors are stored in a
FAISS index (using cosine similarity), while a lexical index is
maintained via scikit‑learn's ``TfidfVectorizer``.  Both the vector
index and associated metadata are persisted to disk under
``settings.faiss_dir``.

The index tracks three parallel lists:

``documents``
    High level metadata about each ingested document (title, source
    URL, sha256, etc.).

``chunk_metadata``
    Metadata for each chunk, including the originating document ID,
    chunk ID, title, source URL, and page numbers.

``chunk_texts``
    The raw text of each chunk.  This is used to build the TF‑IDF
    matrix for lexical retrieval.

Loading the index from disk will rebuild the TF‑IDF matrix on start
up.  When new documents are added the FAISS index is updated and the
lexical model is retrained.  To achieve deterministic performance the
index is saved back to disk after each ingestion.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from config import settings
import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:
    raise ImportError(
        "faiss-cpu is required for the FAISS index. Please add faiss-cpu to your requirements and install it."
    ) from exc

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as exc:
    raise ImportError(
        "scikit-learn is required for TF-IDF retrieval. Please add scikit-learn to your requirements and install it."
    ) from exc


class IndexStore:
    """Hybrid dense/lexical index persisted to disk."""

    def __init__(self, dir_path: str, embed_dim: int, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.dir = dir_path
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.documents: List[Dict[str, Any]] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.chunk_texts: List[str] = []
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any | None = None
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    #
    def load(self) -> None:
        """Load the FAISS index and metadata from disk, if present."""
        os.makedirs(self.dir, exist_ok=True)
        index_path = os.path.join(self.dir, "faiss.index")
        meta_path = os.path.join(self.dir, "metadata.json")
        # Load or initialise FAISS index
        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)
        else:
            # Use IndexFlatIP for cosine similarity (vectors are normalised)
            self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
        # Load metadata
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.chunk_metadata = data.get("chunk_metadata", [])
            self.chunk_texts = data.get("chunk_texts", [])
        # Build TF‑IDF model if we have any text
        if self.chunk_texts:
            self._build_tfidf()

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self.faiss_index is None:
            return
        os.makedirs(self.dir, exist_ok=True)
        # Save FAISS index
        faiss.write_index(self.faiss_index, os.path.join(self.dir, "faiss.index"))
        # Persist metadata
        with open(os.path.join(self.dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": self.documents,
                    "chunk_metadata": self.chunk_metadata,
                    "chunk_texts": self.chunk_texts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ------------------------------------------------------------------
    # TF‑IDF construction
    #
    def _build_tfidf(self) -> None:
        """(Re)build the TF‑IDF model from current chunk texts."""
        # Build vectorizer on unigrams
        self.tfidf_vectorizer = TfidfVectorizer()
        # Fit transform returns a sparse matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunk_texts)

    # ------------------------------------------------------------------
    # Document and chunk ingestion
    #
    def add_documents(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        chunk_meta: List[Dict[str, Any]],
        doc_meta: Dict[str, Any],
    ) -> None:
        """
        Add new document chunks to the index.

        Parameters
        ----------
        embeddings : List of dense embeddings (lists of floats).
        texts : List of chunk texts corresponding to the embeddings.
        chunk_meta : List of metadata dicts for each chunk.  Must be the same
            length as ``embeddings`` and ``texts``.
        doc_meta : Metadata dict describing the entire document.
        """
        if len(embeddings) != len(texts) or len(texts) != len(chunk_meta):
            raise ValueError("embeddings, texts, and chunk_meta must have the same length")
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not initialised")
        # Append document metadata
        self.documents.append(doc_meta)
        # Append chunk metadata and texts
        self.chunk_metadata.extend(chunk_meta)
        self.chunk_texts.extend(texts)
        # Prepare dense vectors
        arr = np.vstack(embeddings).astype("float32")
        # Normalise vectors for cosine similarity
        faiss.normalize_L2(arr)
        # Add to FAISS index
        self.faiss_index.add(arr)
        # Rebuild TF‑IDF for lexical retrieval
        self._build_tfidf()
        # Persist changes
        self.save()

    # ------------------------------------------------------------------
    # Search
    #


    def search(self, query_vector: List[float], query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top K chunks relevant to the query using hybrid search.

        A dense search is performed using the FAISS index (cosine similarity).
        A lexical search is performed using the TF-IDF model. Scores are
        normalized and blended according to self.alpha.

        Returns a list of chunk metadata dicts ordered by combined score.
        """
        if self.faiss_index is None or not self.chunk_metadata:
            return []

        # Dense search across all vectors
        q = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(q)
        dists, idxs = self.faiss_index.search(q, len(self.chunk_metadata))
        dense_scores = dists[0]

        # Lexical search
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            lexical_scores = np.zeros(len(self.chunk_metadata), dtype="float32")
        else:
            q_vec = self.tfidf_vectorizer.transform([query_text])
            lexical_scores = (q_vec @ self.tfidf_matrix.T).toarray().flatten()

        # Normalize dense scores to [0,1]
        if dense_scores.size > 0:
            d_max, d_min = float(dense_scores.max()), float(dense_scores.min())
            dense_norm = (dense_scores - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(dense_scores)
        else:
            dense_norm = np.zeros(len(self.chunk_metadata), dtype="float32")

        # Normalize lexical scores to [0,1]
        if lexical_scores.size > 0:
            l_max, l_min = float(lexical_scores.max()), float(lexical_scores.min())
            lexical_norm = (lexical_scores - l_min) / (l_max - l_min) if l_max > l_min else np.zeros_like(lexical_scores)
        else:
            lexical_norm = np.zeros(len(self.chunk_metadata), dtype="float32")

        # Combine & threshold
        combined_scores = self.alpha * dense_norm + (1.0 - self.alpha) * lexical_norm

        # Apply min similarity threshold (drop weak matches)
        mask = combined_scores >= float(getattr(settings, "min_sim_threshold", 0.30))
        if mask.sum() == 0:
            # if everything is below threshold, fall back to top_k with combined order
            top_indices = np.argsort(-combined_scores)[: max(top_k, 0)]
            return [self.chunk_metadata[i] for i in top_indices]

        filtered_indices = np.where(mask)[0]
        filtered_scores = combined_scores[mask]

        # Rank filtered by combined score
        order = np.argsort(-filtered_scores)
        filtered_sorted = filtered_indices[order]

        # Cap to top_k
        top_indices = filtered_sorted[: max(top_k, 0)]
        return [self.chunk_metadata[i] for i in top_indices]


    # ------------------------------------------------------------------
    # Metadata helpers
    #
    def get_document_metadata(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping from doc_id to document metadata for the given IDs.
        """
        lookup = {}
        for doc in self.documents:
            if doc.get("doc_id") in doc_ids:
                lookup[doc["doc_id"]] = doc
        return lookup
