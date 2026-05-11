FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=off

WORKDIR /app

COPY requirements.txt ./
RUN set -eux; \
    for attempt in 1 2 3; do \
        python -m pip install --prefer-binary --timeout 120 --retries 10 --progress-bar off -r requirements.txt && break; \
        if [ "$attempt" = "3" ]; then exit 1; fi; \
        sleep 10; \
    done; \
    python -m pip cache purge || true

COPY . .

RUN mkdir -p /app/artifacts/runs

EXPOSE 8000 8501
