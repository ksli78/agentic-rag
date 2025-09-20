FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RETRIEVER=llama \
    MAX_CHUNKS_PER_DOC=3 \
    W_LEX=1 \
    W_KW=1 \
    W_CAT=1 \
    OLLAMA_SEED=42 \
    GEN_TEMPERATURE=0.2 \
    GEN_TOP_P=1.0 \
    GEN_REPEAT_PENALTY=1.0 \
    GEN_NUM_CTX=6

WORKDIR /app

# small toolset
RUN apt-get update && apt-get install -y --no-install-recommends \
      tini curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.txt

COPY . /app/

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
