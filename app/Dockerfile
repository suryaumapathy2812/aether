FROM python:3.12-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── System packages ────────────────────────────────────────────────────────────
# Keep this lean — only packages that are always needed and have no good
# runtime alternative. Everything else (ffmpeg, imagemagick, editors, etc.)
# can be installed on demand via: apt-get install <pkg>
#
# Build deps — required for Python C-extensions + aiortc/av (WebRTC)
# Shell essentials — baseline every AI agent runtime ships (curl, git, jq, rg…)
# Scripting utils — make, sqlite3 (inspect agent DB), file, procps
RUN apt-get update && apt-get install -y --no-install-recommends \
    \
    # Build / C-extension deps \
    gcc \
    build-essential \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libopus-dev \
    libvpx-dev \
    \
    # Shell / text essentials \
    ca-certificates \
    curl \
    wget \
    git \
    jq \
    ripgrep \
    unzip \
    zip \
    less \
    procps \
    file \
    make \
    sqlite3 \
    \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock .python-version README.md ./

# Install dependencies to system Python (including webrtc optional group)
ENV UV_SYSTEM_PYTHON=1
RUN uv sync --frozen --no-dev --no-install-project --extra webrtc

# Copy application code
COPY src/ src/
COPY skills/ skills/
COPY plugins/ plugins/

# Install the project itself to system Python
RUN uv sync --frozen --no-dev --extra webrtc

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

# Use uv run to activate the environment
CMD ["uv", "run", "uvicorn", "aether.main:app", "--host", "0.0.0.0", "--port", "8000"]
