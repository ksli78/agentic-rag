"""Application configuration using environment variables.

The settings defined here control the behaviour of the API service.
Defaults are provided for all options so that the application can run
without a .env file, but any value can be overridden by setting
environment variables.  See ``Settings`` for a description of each
field.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama configuration
    ollama_host: str = "http://ollama:11434"
    ollama_embed_model: str = "embeddinggemma:latest"
    ollama_gen_model: str = "llama3.2:latest"

    # Embeddings
    embed_dim: int = 768

    # Storage directories
    faiss_dir: str = "/data/faiss_index"
    tfidf_dir: str = "/data/faiss_index"
    documents_path: str = "/data/documents.json"
    chunks_path: str = "/data/chunks.json"

    # Chunking and retrieval
    chunk_size: int = 600
    chunk_overlap: int = 100
    min_sim_threshold: float = 0.30
    max_context_docs: int = 3
    max_chunks_per_doc: int = 13
    alpha: float = 0.5  # weight for dense vs lexical in hybrid retrieval

    # LLM generation
    default_temperature: float = 0.2

    # PDF parser
    doc_parse_engine: str = "docling"  # docling | fitz

    # Fallback
    portal_url: str = "https://portal.amentumspacemissions.com/MS/Pages/MSDefaultHomePage.aspx"

    # Logging
    log_level: str = "INFO"


settings = Settings()
