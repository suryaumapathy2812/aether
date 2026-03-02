package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

func HTTPStream(client *http.Client, w http.ResponseWriter, incoming *http.Request, host string, port int, path string, userID string, enforceUser bool) bool {
	ctx := incoming.Context()
	upstreamReq, err := BuildUpstreamRequest(ctx, incoming, host, port, path, userID, enforceUser)
	if err != nil {
		return false
	}
	proxyClient := *client
	proxyClient.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	}
	resp, err := proxyClient.Do(upstreamReq)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	CopyResponseHeaders(w.Header(), resp.Header)
	if strings.HasPrefix(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		w.Header().Set("Cache-Control", "no-cache, no-transform")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
	}
	w.WriteHeader(resp.StatusCode)
	if flusher, ok := w.(http.Flusher); ok {
		buf := make([]byte, 32*1024)
		for {
			n, readErr := resp.Body.Read(buf)
			if n > 0 {
				_, _ = w.Write(buf[:n])
				flusher.Flush()
			}
			if readErr != nil {
				break
			}
		}
	} else {
		_, _ = io.Copy(w, resp.Body)
	}
	return true
}

func BuildUpstreamRequest(ctx context.Context, incoming *http.Request, host string, port int, path string, userID string, enforceUser bool) (*http.Request, error) {
	u := &url.URL{
		Scheme:   "http",
		Host:     fmt.Sprintf("%s:%d", host, port),
		Path:     path,
		RawQuery: incoming.URL.RawQuery,
	}
	if enforceUser {
		u.RawQuery = RewriteQueryUserID(path, u.Query(), userID).Encode()
	}

	bodyBytes := []byte(nil)
	if incoming.Body != nil {
		b, err := io.ReadAll(incoming.Body)
		if err != nil {
			return nil, err
		}
		bodyBytes = b
	}
	if enforceUser {
		bodyBytes = RewriteBodyUserID(path, incoming.Header.Get("Content-Type"), bodyBytes, userID)
	}

	var bodyReader io.Reader
	if len(bodyBytes) > 0 {
		bodyReader = bytes.NewReader(bodyBytes)
	}
	req, err := http.NewRequestWithContext(ctx, incoming.Method, u.String(), bodyReader)
	if err != nil {
		return nil, err
	}
	CopyRequestHeaders(req.Header, incoming.Header)
	return req, nil
}

func RewriteQueryUserID(path string, q url.Values, userID string) url.Values {
	copyQ := url.Values{}
	for k, vals := range q {
		copyQ[k] = append([]string(nil), vals...)
	}
	if strings.HasPrefix(path, "/api/memory/") || strings.HasPrefix(path, "/v1/agent/tasks") {
		copyQ.Set("user_id", userID)
	}
	return copyQ
}

func RewriteBodyUserID(path, contentType string, body []byte, userID string) []byte {
	if len(body) == 0 {
		return body
	}
	if !strings.Contains(strings.ToLower(contentType), "application/json") {
		return body
	}
	if !strings.HasPrefix(path, "/v1/") && !strings.HasPrefix(path, "/api/") {
		return body
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return body
	}
	payload["user_id"] = userID
	b, err := json.Marshal(payload)
	if err != nil {
		return body
	}
	return b
}

func CopyRequestHeaders(dst, src http.Header) {
	for k, vals := range src {
		kl := strings.ToLower(k)
		if kl == "host" || kl == "content-length" || kl == "connection" || kl == "upgrade" {
			continue
		}
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}

func CopyResponseHeaders(dst, src http.Header) {
	for k, vals := range src {
		kl := strings.ToLower(k)
		if kl == "connection" || kl == "transfer-encoding" || kl == "keep-alive" || kl == "upgrade" {
			continue
		}
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}
