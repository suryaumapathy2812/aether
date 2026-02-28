FROM golang:1.24-alpine

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o /usr/local/bin/orchestrator-go ./cmd/server

EXPOSE 9000

CMD ["orchestrator-go"]
