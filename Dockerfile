# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps needed by ChromaDB / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: dependencies ──────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Stage 3: final image ───────────────────────────────────────────────────────
FROM deps AS final

COPY . /app

# Create data directories
RUN mkdir -p /app/chroma_data /app/logs

# Expose API + frontend ports
EXPOSE 8000 8501

# Default: run the API (override in docker-compose for frontend)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
