import os
import lancedb
from typing import Tuple
from config import settings
from models import Document, Chunk




def get_db() -> lancedb.DBConnection:
    os.makedirs(settings.lancedb_uri, exist_ok=True)
    return lancedb.connect(settings.lancedb_uri)


def get_or_create_tables() -> Tuple[lancedb.table, lancedb.table]:
    db = get_db()

    # Create/open tables
    if "documents" in db.table_names():
        docs = db.open_table("documents")
    else:
        docs = db.create_table("documents", schema=Document.to_arrow_schema())

    if "chunks" in db.table_names():
        chunks = db.open_table("chunks")
    else:
        chunks = db.create_table("chunks", schema=Chunk.to_arrow_schema())

    # Try to create an ANN index on the embedding column (ignore if unsupported/existing)
    try:
        # Newer LanceDB signatures:
        #   create_index(vector_column_name="embedding", index_type="IVF_HNSW_PQ", metric="cosine")
        # Older/alt signatures also exist; this call will no-op if it already exists or raise.
        chunks.create_index(
            vector_column_name="embedding",
            index_type="IVF_HNSW_PQ",
            metric="cosine",
        )
    except Exception:
        # Either already indexed or the API isn't available in this version; continue without failing.
        pass

    return docs, chunks