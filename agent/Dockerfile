# ── Build stage ──
FROM golang:1.24-bookworm AS builder

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_TIME=unknown

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config ca-certificates && rm -rf /var/lib/apt/lists/*

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=1 go build \
  -ldflags "-s -w \
    -X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.Version=${VERSION} \
    -X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.Commit=${COMMIT} \
    -X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.BuildTime=${BUILD_TIME}" \
  -o /agent-server ./cmd/server

# ── Runtime stage ──
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    curl \
    jq \
    python3 \
    python3-pip \
    nodejs \
    npm \
    git \
    wget \
    unzip \
    zip \
    bash \
    coreutils \
    fuse3 \
    && rm -rf /var/lib/apt/lists/*

# Install rclone
RUN curl -fsSL https://rclone.org/install.sh | bash

# FUSE config — allow --allow-other flag
RUN echo "user_allow_other" >> /etc/fuse.conf

# Common Python packages
RUN pip3 install requests beautifulsoup4 pandas python-docx openpyxl PyPDF2 --break-system-packages --no-warn-script-location

# Common Node packages
RUN npm install -g axios

# Create workspace and mount directories
RUN mkdir -p /workspace/gdrive /root/.config/rclone

COPY --from=builder /agent-server /usr/local/bin/agent-server
COPY --from=builder /app/assets /app/assets

WORKDIR /workspace
EXPOSE 8000

CMD ["agent-server"]
