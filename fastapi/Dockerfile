FROM python:3.11-slim

# Install system dependencies needed by docling and PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Define requirements inline to avoid extra files
COPY <<'PIP' /app/requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.4.0
httpx==0.27.2
numpy==1.26.4
scikit-learn==1.4.2
faiss-cpu==1.8.0
# Docling for PDF parsing
docling==2.15.0
docling-core==2.15.0
# PyMuPDF as a fallback PDF parser
pymupdf==1.23.24
# Required for UploadFile/Form
python-multipart==0.0.9
sentence-transformers==2.6.1
PIP

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Create data directory for FAISS index
RUN mkdir -p /data/faiss_index

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
