package observability

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/tracelog"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

type ctxKey string

const requestIDKey ctxKey = "request_id"

var (
	forensicMode  atomic.Bool
	verboseMode   atomic.Bool
	bodyLimitByte atomic.Int64
)

func Init(service string) {
	if strings.TrimSpace(service) == "" {
		service = "service"
	}
	level := parseLevel(strings.TrimSpace(os.Getenv("LOG_LEVEL")))
	zerolog.SetGlobalLevel(level)
	if strings.EqualFold(strings.TrimSpace(os.Getenv("LOG_FORMAT")), "text") {
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
	verbose := false
	if raw := strings.TrimSpace(os.Getenv("LOG_VERBOSE_MODE")); raw != "" {
		if v, err := strconv.ParseBool(raw); err == nil {
			verbose = v
		}
	}
	verboseMode.Store(verbose)
	limit := int64(1 * 1024 * 1024)
	if raw := strings.TrimSpace(os.Getenv("LOG_BODY_MAX_BYTES")); raw != "" {
		if n, err := strconv.ParseInt(raw, 10, 64); err == nil && n > 0 {
			limit = n
		}
	}
	bodyLimitByte.Store(limit)
	log.Info().Str("event", "logger_initialized").Str("level", level.String()).Bool("forensic_mode", forensic).Bool("verbose_mode", verboseMode.Load()).Int64("body_max_bytes", limit).Msg("logger ready")
}

func VerboseEnabled() bool {
	return verboseMode.Load()
}

func ShouldLogNoisySQL(sql string, hasErr bool) bool {
	if hasErr {
		return true
	}
	if VerboseEnabled() {
		return true
	}
	q := strings.ToUpper(strings.TrimSpace(sql))
	switch q {
	case "BEGIN", "COMMIT", "ROLLBACK":
		return false
	}
	if strings.HasPrefix(q, "SAVEPOINT") || strings.HasPrefix(q, "RELEASE SAVEPOINT") {
		return false
	}
	return true
}

func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		requestID := strings.TrimSpace(r.Header.Get("X-Request-ID"))
		if requestID == "" {
			requestID = newID()
		}
		r = r.WithContext(context.WithValue(r.Context(), requestIDKey, requestID))

		body := ""
		if forensicMode.Load() {
			body = readAndRestoreBody(&r.Body)
		}
		captureBody := forensicMode.Load() && !strings.HasPrefix(r.Header.Get("Content-Type"), "text/event-stream")
		rw := &responseLogger{ResponseWriter: w, status: http.StatusOK, captureBody: captureBody}
		rw.Header().Set("X-Request-ID", requestID)

		in := log.Info().Str("event", "http_request_in").Str("request_id", requestID).Str("method", r.Method).Str("path", r.URL.Path).Str("query", r.URL.RawQuery).Str("remote_addr", r.RemoteAddr).Interface("headers", sanitizeHeaders(r.Header, forensicMode.Load()))
		if body != "" {
			in.Str("body", body)
		}
		in.Msg("request received")

		next.ServeHTTP(rw, r)

		out := log.Info().Str("event", "http_request_out").Str("request_id", requestID).Str("method", r.Method).Str("path", r.URL.Path).Int("status", rw.status).Int("bytes", rw.bytes).Dur("duration", time.Since(start))
		if rw.captureBody && rw.body.Len() > 0 {
			out.Str("response_body", truncate(rw.body.String(), int(bodyLimitByte.Load())))
		}
		out.Msg("request completed")
	})
}

func WrapTransport(base http.RoundTripper) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	return &transportLogger{base: base}
}

func RequestID(ctx context.Context) string {
	v, _ := ctx.Value(requestIDKey).(string)
	return v
}

func PGXTraceLogger() *tracelog.TraceLog {
	return &tracelog.TraceLog{Logger: pgxLogger{}, LogLevel: tracelog.LogLevelTrace}
}

type pgxLogger struct{}

func (pgxLogger) Log(ctx context.Context, level tracelog.LogLevel, msg string, data map[string]interface{}) {
	hasErr := level == tracelog.LogLevelError
	sqlText := firstSQLValue(data)
	if !ShouldLogNoisySQL(sqlText, hasErr) {
		return
	}
	e := log.Debug().Str("event", "pgx_query").Str("level", level.String()).Str("msg", msg)
	if reqID := RequestID(ctx); reqID != "" {
		e.Str("request_id", reqID)
	}
	if data != nil {
		e.Interface("data", data)
	}
	e.Msg("pgx trace")
}

type responseLogger struct {
	http.ResponseWriter
	status      int
	bytes       int
	body        bytes.Buffer
	captureBody bool
}

func (l *responseLogger) WriteHeader(statusCode int) {
	l.status = statusCode
	l.ResponseWriter.WriteHeader(statusCode)
}

func (l *responseLogger) Write(p []byte) (int, error) {
	if l.captureBody && int64(l.body.Len()) < bodyLimitByte.Load() {
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
	start := time.Now()
	reqID := strings.TrimSpace(req.Header.Get("X-Request-ID"))
	if reqID == "" {
		reqID = RequestID(req.Context())
	}
	if reqID == "" {
		reqID = newID()
	}
	req.Header.Set("X-Request-ID", reqID)

	var body string
	if forensicMode.Load() {
		body = readRequestBodyForReplay(req)
	}
	in := log.Debug().Str("event", "http_client_request").Str("request_id", reqID).Str("method", req.Method).Str("url", req.URL.String()).Interface("headers", sanitizeHeaders(req.Header, forensicMode.Load()))
	if body != "" {
		in.Str("body", body)
	}
	in.Msg("outbound request")

	resp, err := t.base.RoundTrip(req)
	if err != nil {
		log.Error().Str("event", "http_client_error").Str("request_id", reqID).Str("method", req.Method).Str("url", req.URL.String()).Dur("duration", time.Since(start)).Err(err).Msg("outbound request failed")
		return nil, err
	}
	out := log.Debug().Str("event", "http_client_response").Str("request_id", reqID).Str("method", req.Method).Str("url", req.URL.String()).Int("status", resp.StatusCode).Dur("duration", time.Since(start)).Interface("headers", sanitizeHeaders(resp.Header, forensicMode.Load()))
	if forensicMode.Load() {
		respBody := readAndRestoreBody(&resp.Body)
		if respBody != "" {
			out.Str("body", respBody)
		}
	}
	out.Msg("outbound response")
	return resp, nil
}

func parseLevel(raw string) zerolog.Level {
	raw = strings.ToLower(strings.TrimSpace(raw))
	if raw == "verbose" {
		return zerolog.DebugLevel
	}
	if raw == "" {
		return zerolog.DebugLevel
	}
	if lvl, err := zerolog.ParseLevel(raw); err == nil {
		return lvl
	}
	return zerolog.DebugLevel
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

func firstSQLValue(data map[string]interface{}) string {
	if data == nil {
		return ""
	}
	for _, key := range []string{"sql", "query", "stmt", "statement"} {
		if v, ok := data[key]; ok {
			return strings.TrimSpace(toString(v))
		}
	}
	return ""
}

func toString(v interface{}) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return strings.TrimSpace(fmt.Sprintf("%v", v))
}
