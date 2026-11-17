# Small production image. The corpus, store, and results are mounted, not baked in.
# Build:  docker build -t nifty-rag .
# Run:    docker run --rm -p 8000:8000 --env-file .env \
#            -v "$PWD/.pageindex:/data/.pageindex:ro" nifty-rag
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    PAGEINDEX_STORE=/data/.pageindex

WORKDIR /app

RUN adduser --disabled-password --gecos "" --uid 10001 app
RUN mkdir -p /data && chown app:app /data
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py index.html index_pdfs.py ./
RUN chown -R app:app /app
USER app

EXPOSE 8000
VOLUME ["/data"]

CMD ["python", "server.py"]
