docker build --no-cache -t agentic-rag-api:v1.1.1.17 -f fastapi/Dockerfile fastapi
docker save agentic-rag-api:v1.1.1.17 -o agentic-rag-api_v1.1.1.17.tar
