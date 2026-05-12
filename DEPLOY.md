# Deploy branch over SSH

Runbook for deploying the `deploy` branch to a remote Linux server with Docker Compose.

The server setup exposes only Streamlit UI on port `8501`. FastAPI backend port `8000` stays private inside the Docker network.

## 0. Variables

On your laptop:

```bash
SERVER=user@SERVER_IP
APP_DIR=~/ECOS-AI-Assistant
```

Inside the SSH session on the server:

```bash
APP_DIR=~/ECOS-AI-Assistant
REPO_URL=<git-repo-url>
BRANCH=deploy
```

## 1. Server prerequisites

SSH into the server:

```bash
ssh "$SERVER"
```

Install Docker Engine and the Docker Compose plugin if they are missing:

```bash
docker --version
docker compose version
```

If those commands fail, install Docker first using your server OS instructions.

Open the UI port in the firewall/security group:

```bash
sudo ufw allow 8501/tcp
```

Keep port `8000` closed unless you intentionally want public API access.

## 2. First deploy from Git

On the server:

```bash
APP_DIR=~/ECOS-AI-Assistant
REPO_URL=<git-repo-url>
BRANCH=deploy
git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"
mkdir -p artifacts
cp .env.example .env
nano .env
```

Fill `.env` with real API keys. Do not commit `.env`.

If the server needs local catalog data, upload it from your laptop:

```bash
rsync -av ./data/ "$SERVER:$APP_DIR/data/"
```

Build and start:

```bash
docker compose -f docker-compose.server.yml build
docker compose -f docker-compose.server.yml up -d
```

Check:

```bash
docker compose -f docker-compose.server.yml ps
docker compose -f docker-compose.server.yml logs -f backend
```

Open:

```text
http://SERVER_IP:8501
```

## 3. If the branch is not in Git yet

Upload current files from your laptop instead of cloning:

```bash
rsync -av \
  --exclude .git \
  --exclude .venv \
  --exclude artifacts \
  --exclude dumps \
  ./ "$SERVER:$APP_DIR/"
```

Then on the server:

```bash
APP_DIR=~/ECOS-AI-Assistant
cd "$APP_DIR"
mkdir -p artifacts
cp .env.example .env
nano .env
docker compose -f docker-compose.server.yml up -d --build
```

## 4. Update existing deploy

On the server:

```bash
APP_DIR=~/ECOS-AI-Assistant
BRANCH=deploy
cd "$APP_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
docker compose -f docker-compose.server.yml up -d --build
```

For logs:

```bash
docker compose -f docker-compose.server.yml logs -f ui
docker compose -f docker-compose.server.yml logs -f backend
```

## 5. Useful operations

Restart without rebuild:

```bash
docker compose -f docker-compose.server.yml restart
```

Stop:

```bash
docker compose -f docker-compose.server.yml down
```

Check backend from inside the server:

```bash
docker compose -f docker-compose.server.yml exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

Show images:

```bash
docker images | grep ecos-ai-assistant
```

## 6. Notes

- `docker-compose.server.yml` uses the lightweight backend runtime dependencies from `requirements-backend.txt`.
- The lightweight backend uses BM25 retrieval. It does not install Chroma, LlamaIndex, HuggingFace embeddings, or Torch.
- Use `requirements-vector.txt` only on a separate indexing machine/job if you need to rebuild the Chroma vector index.
- `data/` is mounted read-only into the backend. `artifacts/` is mounted writable.
- Each browser gets its own Streamlit session; backend runs create separate `run_id` artifact folders.
- Do not paste `docker compose config` output publicly: it expands `.env` values.
