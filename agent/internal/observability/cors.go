package observability

import (
	"net/http"
	"strconv"
	"strings"
)

// CORSConfig controls which cross-origin requests the server will accept.
type CORSConfig struct {
	AllowedOrigins   []string
	AllowedMethods   []string
	AllowedHeaders   []string
	ExposeHeaders    []string
	MaxAge           int
	AllowCredentials bool
}

// CORSMiddleware returns an HTTP middleware that sets CORS response headers
// based on the provided configuration.  It handles OPTIONS preflight requests
// by responding with 204 No Content and the appropriate headers, without
// forwarding the request to the next handler.
//
// Security notes:
//   - The wildcard origin "*" must NOT be combined with AllowCredentials=true
//     (browsers reject this combination).
//   - The Vary: Origin header is always set to prevent cache poisoning when
//     the server reflects a specific origin.
func CORSMiddleware(cfg CORSConfig) func(http.Handler) http.Handler {
	// Pre-compute joined header values so we don't allocate on every request.
	methodsHeader := strings.Join(cfg.AllowedMethods, ", ")
	headersHeader := strings.Join(cfg.AllowedHeaders, ", ")
	exposeHeader := strings.Join(cfg.ExposeHeaders, ", ")
	maxAgeHeader := strconv.Itoa(cfg.MaxAge)

	// Build a set for O(1) origin lookups.
	allowAll := false
	originSet := make(map[string]struct{}, len(cfg.AllowedOrigins))
	for _, o := range cfg.AllowedOrigins {
		if o == "*" {
			allowAll = true
			break
		}
		originSet[o] = struct{}{}
	}

	// If wildcard + credentials are both set, disable credentials to comply
	// with the CORS spec (browsers will reject the response anyway).
	allowCredentials := cfg.AllowCredentials
	if allowAll && allowCredentials {
		allowCredentials = false
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := strings.TrimSpace(r.Header.Get("Origin"))

			// Always set Vary: Origin for correct caching behaviour.
			w.Header().Add("Vary", "Origin")

			// Determine whether this origin is allowed.
			var allowed bool
			var allowOriginValue string

			if origin == "" {
				// Same-origin or non-browser request — no CORS headers needed.
				next.ServeHTTP(w, r)
				return
			}

			if allowAll {
				allowed = true
				allowOriginValue = "*"
			} else {
				_, found := originSet[origin]
				if found {
					allowed = true
					allowOriginValue = origin // reflect the specific origin
				}
			}

			if !allowed {
				// Origin not in the allow-list — serve the request normally
				// but without any CORS headers so the browser blocks it.
				next.ServeHTTP(w, r)
				return
			}

			// ── Set CORS response headers ──────────────────────────────
			h := w.Header()
			h.Set("Access-Control-Allow-Origin", allowOriginValue)
			h.Set("Access-Control-Allow-Methods", methodsHeader)
			h.Set("Access-Control-Allow-Headers", headersHeader)
			if exposeHeader != "" {
				h.Set("Access-Control-Expose-Headers", exposeHeader)
			}
			if maxAgeHeader != "0" {
				h.Set("Access-Control-Max-Age", maxAgeHeader)
			}
			if allowCredentials {
				h.Set("Access-Control-Allow-Credentials", "true")
			}

			// ── Preflight (OPTIONS) — respond immediately ──────────────
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}
