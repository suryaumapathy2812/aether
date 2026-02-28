package logic

import (
	"context"
	"net/http"
	"time"
)

type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

func defaultHTTPClient(c HTTPClient) HTTPClient {
	if c != nil {
		return c
	}
	return &http.Client{Timeout: 20 * time.Second}
}

type WebResult struct {
	Title       string
	URL         string
	Description string
}

type NewsResult struct {
	Title       string
	URL         string
	Source      string
	Age         string
	Description string
}

type ContextSnippet struct {
	Title string
	URL   string
	Text  string
}

type ServiceCall[T any] interface {
	Execute(ctx context.Context) (T, error)
}
