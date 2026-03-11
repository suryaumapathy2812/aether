FROM golang:1.24-alpine

RUN apk add --no-cache gcc musl-dev sqlite-dev

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_TIME=unknown

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -ldflags "-X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.Version=${VERSION} -X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.Commit=${COMMIT} -X github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo.BuildTime=${BUILD_TIME}" -o /usr/local/bin/agent-server ./cmd/server

EXPOSE 8000

CMD ["agent-server"]
