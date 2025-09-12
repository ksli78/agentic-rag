# --- Base: lean Python; CUDA comes via PyTorch cu121 wheels ---
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Small set of helpful tools
RUN apt-get update && apt-get install -y --no-install-recommends \
      tini curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Requirements first (context is already fastapi/) ----
# If your build context is fastapi/, requirements.txt is at the root of the context
COPY requirements.txt /app/requirements.txt

# Install CUDA-enabled PyTorch first (no torch in requirements.txt)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --index-url ${TORCH_INDEX_URL} \
      torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Then install the rest (this will bring in marker‑pdf 1.6.0 and transformers 4.45.2)
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Re-assert CUDA torchvision to ensure no later step replaced it
RUN pip install --no-deps --force-reinstall \
    --index-url ${TORCH_INDEX_URL} \
    "torchvision==0.18.1"


# ---- App code ----
# IMPORTANT: because the build context is fastapi/, COPY . /app/ copies your main.py, etc.
COPY . /app/

# (Optional) Warm Marker model caches into the image to help offline/air-gapped use.
# Uncomment if you want to bake weights into the image:
#RUN python - <<'PY'
#from marker.models import create_model_dict
#create_model_dict(device="cuda")
#print("Marker models cached (cuda)")
#PY

# ---- Runtime env (override in compose if you like) ----
ENV OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_EMBED_MODEL=embeddinggemma:latest \
    OLLAMA_GEN_MODEL=llama3.2:latest \
    USE_MARKER=true \
    MARKER_USE_LLM=true \
    MARKER_OLLAMA_MODEL=llama3:8b \
    MARKER_DEVICE=cuda \
    CHUNK_SIZE=1200 \
    CHUNK_OVERLAP=200

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
