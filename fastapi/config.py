from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    ollama_host: str = "http://ollama:11434"
    ollama_embed_model: str = "embeddinggemma:latest"
    ollama_gen_model: str = "llama3.2:latest"

    lancedb_uri: str = "/data/lancedb"
    embed_dim: int = 768

    chunk_size: int = 1200
    chunk_overlap: int = 200

    min_sim_threshold: float = 0.30
    max_context_docs: int = 3

    default_temperature: float = 0.2

    portal_url: str = "https://portal.amentumspacemissions.com/MS/Pages/MSDefaultHomePage.aspx"

settings = Settings()
