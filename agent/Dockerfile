FROM golang:1.24-alpine

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o /usr/local/bin/agent-server ./cmd/server

EXPOSE 8000

CMD ["agent-server"]
