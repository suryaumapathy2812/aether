FROM python:3.12-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock .python-version README.md ./

# Install dependencies to system Python
ENV UV_SYSTEM_PYTHON=1
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ src/

# Install the project itself to system Python
RUN uv sync --frozen --no-dev

ENV AETHER_HOST=0.0.0.0 \
    AETHER_PORT=9000

EXPOSE 9000

# Use uv run to activate the environment
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9000"]
