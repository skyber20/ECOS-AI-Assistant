#!/bin/sh
set -eu

BACKEND_PORT="${BACKEND_PORT:-8000}"
PUBLIC_PORT="${PORT:-8501}"

export BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"

uvicorn app.api:app --host 127.0.0.1 --port "${BACKEND_PORT}" &
BACKEND_PID="$!"

trap 'kill "${BACKEND_PID}" 2>/dev/null || true' INT TERM EXIT

exec streamlit run app/ui.py \
  --server.address=0.0.0.0 \
  --server.port="${PUBLIC_PORT}" \
  --server.headless=true
