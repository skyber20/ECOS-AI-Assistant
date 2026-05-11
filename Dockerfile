FROM python:3.12-slim

ARG INSTALL_VECTOR_DEPS=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=off

WORKDIR /app

COPY requirements.txt requirements-vector.txt ./
RUN set -eux; \
    install_requirements() { \
        file="$1"; \
        for attempt in 1 2 3; do \
            python -m pip install --prefer-binary --timeout 120 --retries 10 --progress-bar off -r "$file" && return 0; \
            if [ "$attempt" = "3" ]; then return 1; fi; \
            echo "pip install $file failed, retrying in 10s..."; \
            sleep 10; \
        done; \
    }; \
    install_requirements requirements.txt; \
    if [ "$INSTALL_VECTOR_DEPS" = "true" ]; then \
        install_requirements requirements-vector.txt; \
    fi; \
    python -m pip cache purge || true

COPY . .

RUN mkdir -p /app/artifacts/runs

EXPOSE 8000 8501
