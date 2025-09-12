docker build -t agentic-rag-api:v1.1.1.13 -f fastapi/Dockerfile fastapi
docker save agentic-rag-api:v1.1.1.13 -o agentic-rag-api_v1.1.1.13.tar
