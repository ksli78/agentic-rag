docker build -t agentic-rag-api:v1.1.1.8 -f fastapi/Dockerfile fastapi
docker save agentic-rag-api:v1.1.1.8 -o agentic-rag-api_v1.1.1.8.tar
