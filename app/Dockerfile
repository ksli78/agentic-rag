FROM python:3.11-slim

# Avoid interactive tzdata etc.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (curl for quick health checks if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# The app code
COPY agentic_rag_embeddinggemma.py .

# Streamlit defaults
EXPOSE 8501

# Point the app at the Ollama service name inside the Compose network
ENV OLLAMA_BASE_URL=http://ollama:11434

# Launch streamlit
CMD ["streamlit", "run", "agentic_rag_embeddinggemma.py", "--server.address=0.0.0.0", "--server.port=8501"]
