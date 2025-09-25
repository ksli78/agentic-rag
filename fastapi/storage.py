"""Singleton instance of the persistent index store.

This module instantiates the ``IndexStore`` with configuration values
from ``config.settings``.  Importing from this module guarantees
that all parts of the application operate on the same underlying
index.  The store persists dense vectors and metadata to disk and
manages both dense and lexical retrieval.
"""

from __future__ import annotations

from config import settings
from index_store import IndexStore

# Instantiate a single IndexStore based on configuration.  This
# instance is created when the module is first imported and reused
# throughout the application.
store = IndexStore(settings.faiss_dir, settings.embed_dim, settings.alpha)
