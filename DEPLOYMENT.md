# ECOS AI Assistant deployment

## Local or VPS run

1. Create `.env` from `.env.example` and fill provider keys.
2. Make sure Docker can reach Docker Hub and PyPI.
3. Build and start the stack:

```bash
docker compose up --build -d
```

UI: `http://localhost:8501`
Backend healthcheck: `http://localhost:8000/health`

## Runtime dependencies

The default Docker image installs only runtime dependencies for FastAPI,
Streamlit, LLM calls, parquet reading, and BM25 retrieval.

Chroma/HuggingFace vector-index packages are intentionally optional because
they pull `sentence-transformers`, `torch`, and large CUDA wheels on Linux.
If the Chroma index or embedding model is absent, the assistant falls back to
BM25 search.

To install vector-index dependencies in the image:

```bash
docker compose build --build-arg INSTALL_VECTOR_DEPS=true
docker compose up -d
```

## Docker network failures

Errors like these are network/DNS problems, not Python dependency conflicts:

- `lookup registry-1.docker.io: no such host`
- `lookup auth.docker.io: no such host`
- `lookup pypi.org: no such host`
- `Connection broken: IncompleteRead`
- `THESE PACKAGES DO NOT MATCH THE HASHES`

Useful checks:

```bash
docker pull python:3.12-slim
docker compose build backend
```

If Docker Desktop or the VPS cannot resolve Docker Hub, configure Docker DNS
or a working proxy/mirror, then retry the build.
