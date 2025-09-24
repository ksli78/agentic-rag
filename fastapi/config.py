# fastapi/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # Ollama
    ollama_host: str = "http://ollama:11434"
    ollama_embed_model: str = "embeddinggemma:latest"
    ollama_gen_model: str = "llama3.2:latest"

    # Embeddings (Ollama only; EMBED_DIM comes from .env)
    embed_dim: int = 768

    # Vector store / chunking
    lancedb_uri: str = "/data/lancedb"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_sim_threshold: float = 0.30
    max_context_docs: int = 3

    # LLM generation
    default_temperature: float = 0.2

    # PDF parser
    doc_parse_engine: str = "docling"   # docling | fitz

    # Fallback
    portal_url: str = "https://portal.amentumspacemissions.com/MS/Pages/MSDefaultHomePage.aspx"

    # Logging
    log_level: str = "INFO"             # DEBUG|INFO|WARNING|ERROR

settings = Settings()
