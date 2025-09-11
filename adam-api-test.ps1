docker build -t agentic-rag-api:v1.1.1 -f fastapi/Dockerfile fastapi

# Create/ensure a LanceDB folder (safe to re-run)
$OUT = (Get-Location).Path
mkdir -Force "$OUT\local-lancedb" | Out-Null

# Start the container: map host port 8001 to container port 8000
docker run --name adam-api-test --rm -p 8001:8000 `
  -e OLLAMA_HOST=http://host.docker.internal:11434 `
  -e OLLAMA_EMBED_MODEL=embeddinggemma:latest `
  -e OLLAMA_GEN_MODEL=llama3.2:latest `
  -e LANCEDB_URI=/data/lancedb `
  -v "$OUT\local-lancedb:/data/lancedb" `
  agentic-rag-api:v1.1.1

