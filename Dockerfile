FROM python:3.11-slim

# onnxruntime needs libgomp; nothing else is pulled in.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/rag_index.py src/rag_rerank.py ./
COPY src/rag_server.py ./rag_server.py
COPY synonyms.example.json ./synonyms.json

# The reranker weights are NOT baked in: 118 MB of ONNX does not belong in a
# git repository. Run export_reranker.py once and mount the result at
# /app/reranker, or leave it absent -- the service starts either way and
# reports "reranker": false on /health.
ENV RERANK_MODEL_DIR=/app/reranker

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4).status==200 else 1)"

CMD ["python", "-u", "rag_server.py"]
