# ── Build stage ──
FROM golang:1.24-alpine AS builder

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_TIME=unknown

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 go build \
  -ldflags "-s -w \
    -X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.Version=${VERSION} \
    -X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.Commit=${COMMIT} \
    -X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.BuildTime=${BUILD_TIME}" \
  -o /agent-server ./cmd/server

# ── Runtime stage ──
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata

COPY --from=builder /agent-server /usr/local/bin/agent-server
COPY --from=builder /app/assets /app/assets

EXPOSE 8000

CMD ["agent-server"]
