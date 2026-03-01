package observability

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

type ctxKey string

const requestIDKey ctxKey = "request_id"

var (
	serviceName   atomic.Value
	forensicMode  atomic.Bool
	bodyLimitByte atomic.Int64
)

func Init(service string) {
	if strings.TrimSpace(service) == "" {
		service = "service"
	}
	serviceName.Store(service)

	level := parseLevel(strings.TrimSpace(os.Getenv("LOG_LEVEL")))
	zerolog.SetGlobalLevel(level)

	format := strings.ToLower(strings.TrimSpace(os.Getenv("LOG_FORMAT")))
	if format == "text" || format == "console" {
		log.Logger = zerolog.New(zerolog.ConsoleWriter{Out: os.Stdout, TimeFormat: time.RFC3339Nano}).With().Timestamp().Str("service", service).Logger()
	} else {
		log.Logger = zerolog.New(os.Stdout).With().Timestamp().Str("service", service).Logger()
	}

	forensic := level <= zerolog.DebugLevel
	if raw := strings.TrimSpace(os.Getenv("LOG_FORENSIC_MODE")); raw != "" {
		if v, err := strconv.ParseBool(raw); err == nil {
			forensic = v
		}
	}
	forensicMode.Store(forensic)

	limit := int64(1 * 1024 * 1024)
	if raw := strings.TrimSpace(os.Getenv("LOG_BODY_MAX_BYTES")); raw != "" {
		if n, err := strconv.ParseInt(raw, 10, 64); err == nil && n > 0 {
			limit = n
		}
	}
	bodyLimitByte.Store(limit)

	log.Info().Str("event", "logger_initialized").Str("level", level.String()).Bool("forensic_mode", forensicMode.Load()).Int64("body_max_bytes", bodyLimitByte.Load()).Msg("logger ready")
}

func WrapTransport(base http.RoundTripper) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	return &transportLogger{base: base}
}

func Middleware(next http.Handler) http.Handler {
	if next == nil {
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		requestID := strings.TrimSpace(r.Header.Get("X-Request-ID"))
		if requestID == "" {
			requestID = newID()
		}
		r = r.WithContext(context.WithValue(r.Context(), requestIDKey, requestID))

		headers := sanitizeHeaders(r.Header, forensicMode.Load())
		var reqBody string
		if forensicMode.Load() {
			reqBody = readAndRestoreBody(&r.Body)
		}

		rl := &responseLogger{ResponseWriter: w, status: http.StatusOK}
		rl.Header().Set("X-Request-ID", requestID)

		e := log.Info().Str("event", "http_request_in").Str("request_id", requestID).Str("method", r.Method).Str("path", r.URL.Path).Str("query", r.URL.RawQuery).Str("remote_addr", r.RemoteAddr).Interface("headers", headers)
		if reqBody != "" {
			e.Str("body", reqBody)
		}
		e.Msg("request received")

		next.ServeHTTP(rl, r)

		resp := log.Info().Str("event", "http_request_out").Str("request_id", requestID).Str("method", r.Method).Str("path", r.URL.Path).Int("status", rl.status).Int("bytes", rl.bytes).Dur("duration", time.Since(started))
		if forensicMode.Load() && rl.body.Len() > 0 {
			resp.Str("response_body", truncate(rl.body.String(), int(bodyLimitByte.Load())))
		}
		resp.Msg("request completed")
	})
}

func RequestID(ctx context.Context) string {
	v, _ := ctx.Value(requestIDKey).(string)
	return v
}

func parseLevel(raw string) zerolog.Level {
	if raw == "" {
		return zerolog.DebugLevel
	}
	if lvl, err := zerolog.ParseLevel(raw); err == nil {
		return lvl
	}
	return zerolog.DebugLevel
}

type responseLogger struct {
	http.ResponseWriter
	status int
	bytes  int
	body   bytes.Buffer
}

func (l *responseLogger) WriteHeader(statusCode int) {
	l.status = statusCode
	l.ResponseWriter.WriteHeader(statusCode)
}

func (l *responseLogger) Write(p []byte) (int, error) {
	if forensicMode.Load() && int64(l.body.Len()) < bodyLimitByte.Load() {
		_, _ = l.body.Write(p)
	}
	n, err := l.ResponseWriter.Write(p)
	l.bytes += n
	return n, err
}

func (l *responseLogger) Flush() {
	if f, ok := l.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (l *responseLogger) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	h, ok := l.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, http.ErrNotSupported
	}
	return h.Hijack()
}

func (l *responseLogger) Push(target string, opts *http.PushOptions) error {
	p, ok := l.ResponseWriter.(http.Pusher)
	if !ok {
		return http.ErrNotSupported
	}
	return p.Push(target, opts)
}

type transportLogger struct {
	base http.RoundTripper
}

func (t *transportLogger) RoundTrip(req *http.Request) (*http.Response, error) {
	started := time.Now()
	requestID := strings.TrimSpace(req.Header.Get("X-Request-ID"))
	if requestID == "" {
		requestID = RequestID(req.Context())
	}
	if requestID == "" {
		requestID = newID()
	}
	req.Header.Set("X-Request-ID", requestID)

	var reqBody string
	if forensicMode.Load() {
		reqBody = readRequestBodyForReplay(req)
	}

	e := log.Debug().Str("event", "http_client_request").Str("request_id", requestID).Str("method", req.Method).Str("url", req.URL.String()).Interface("headers", sanitizeHeaders(req.Header, forensicMode.Load()))
	if reqBody != "" {
		e.Str("body", reqBody)
	}
	e.Msg("outbound request")

	resp, err := t.base.RoundTrip(req)
	if err != nil {
		log.Error().Str("event", "http_client_error").Str("request_id", requestID).Str("method", req.Method).Str("url", req.URL.String()).Dur("duration", time.Since(started)).Err(err).Msg("outbound request failed")
		return nil, err
	}

	out := log.Debug().Str("event", "http_client_response").Str("request_id", requestID).Str("method", req.Method).Str("url", req.URL.String()).Int("status", resp.StatusCode).Dur("duration", time.Since(started)).Interface("headers", sanitizeHeaders(resp.Header, forensicMode.Load()))
	if forensicMode.Load() {
		respBody := readAndRestoreBody(&resp.Body)
		if respBody != "" {
			out.Str("body", respBody)
		}
	}
	out.Msg("outbound response")
	return resp, nil
}

func sanitizeHeaders(h http.Header, full bool) map[string]string {
	out := map[string]string{}
	for k, vals := range h {
		joined := strings.Join(vals, ",")
		lk := strings.ToLower(strings.TrimSpace(k))
		if !full && (lk == "authorization" || strings.Contains(lk, "token") || strings.Contains(lk, "secret") || strings.Contains(lk, "cookie")) {
			out[k] = "__redacted__"
			continue
		}
		out[k] = joined
	}
	return out
}

func readAndRestoreBody(body *io.ReadCloser) string {
	if body == nil || *body == nil {
		return ""
	}
	b, err := io.ReadAll(io.LimitReader(*body, bodyLimitByte.Load()))
	if err != nil {
		return ""
	}
	_ = (*body).Close()
	*body = io.NopCloser(bytes.NewReader(b))
	return truncate(string(b), int(bodyLimitByte.Load()))
}

func readRequestBodyForReplay(req *http.Request) string {
	if req == nil || req.Body == nil {
		return ""
	}
	if req.GetBody != nil {
		cloned, err := req.GetBody()
		if err == nil {
			defer cloned.Close()
			b, _ := io.ReadAll(io.LimitReader(cloned, bodyLimitByte.Load()))
			return truncate(string(b), int(bodyLimitByte.Load()))
		}
	}
	b, err := io.ReadAll(io.LimitReader(req.Body, bodyLimitByte.Load()))
	if err != nil {
		return ""
	}
	req.Body = io.NopCloser(bytes.NewReader(b))
	return truncate(string(b), int(bodyLimitByte.Load()))
}

func newID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	return hex.EncodeToString(b)
}

func truncate(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max] + "...(truncated)"
}
