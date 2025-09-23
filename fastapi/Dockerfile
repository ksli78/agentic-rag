FROM python:3.11-slim

# (Optional) system deps if needed by docling/pymupdf/poppler; try without first
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements
COPY <<'PIP' /app/requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.4.0
httpx==0.27.2
numpy==1.26.4
lancedb==0.11.0
# Docling
docling==2.15.0
docling-core==2.15.0
# Required for UploadFile/Form
python-multipart==0.0.9
# If you want FAISS optional later:
# faiss-cpu==1.8.0
PIP


RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# create data dir for LanceDB
RUN mkdir -p /data/lancedb

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
