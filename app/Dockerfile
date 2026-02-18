FROM python:3.12-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System deps for numpy/aiosqlite
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock .python-version README.md ./

# Install dependencies (no dev deps in prod)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ src/
COPY skills/ skills/
COPY plugins/ plugins/

# Install the project itself
RUN uv sync --frozen --no-dev

# Create directories for runtime
RUN mkdir -p /data /workspace

# Defaults (override at runtime)
ENV AETHER_HOST=0.0.0.0 \
    AETHER_PORT=8000 \
    AETHER_DB_PATH=/data/aether_memory.db \
    AETHER_WORKING_DIR=/workspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uv", "run", "uvicorn", "aether.main:app", "--host", "0.0.0.0", "--port", "8000"]
