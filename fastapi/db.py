# fastapi/db.py
import os
import logging
import lancedb
from typing import Tuple
from config import settings
from models import Document, Chunk

log = logging.getLogger("api.db")

def get_db() -> lancedb.DBConnection:
    os.makedirs(settings.lancedb_uri, exist_ok=True)
    return lancedb.connect(settings.lancedb_uri)

def get_or_create_tables() -> Tuple[lancedb.table, lancedb.table]:
    db = get_db()

    docs = db.open_table("documents") if "documents" in db.table_names() else db.create_table(
        "documents", schema=Document.to_arrow_schema()
    )
    chunks = db.open_table("chunks") if "chunks" in db.table_names() else db.create_table(
        "chunks", schema=Chunk.to_arrow_schema()
    )

    log.info(f"LanceDB ready. EMBED_DIM={settings.embed_dim}. Docs={docs.count_rows()} Chunks={chunks.count_rows()}")

    # Only attempt index when we actually have vectors
    try:
        n = chunks.count_rows()
        if n and n > 0:
            chunks.create_index(
                vector_column_name="embedding",
                index_type="IVF_HNSW_PQ",
                metric="cosine",
            )
            log.info(f"ANN index ensured on chunks.embedding (rows={n})")
        else:
            log.info("Skipping index creation (no vectors yet).")
    except Exception as e:
        log.info(f"Index creation skipped ({e.__class__.__name__}): {e}")

    return docs, chunks