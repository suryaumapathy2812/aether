// Package httputil provides shared HTTP response helpers used across all
// httpapi handler packages in the agent service.
package httputil

import (
	"encoding/json"
	"net/http"
)

// WriteJSON encodes payload as JSON and writes it with the given status code.
func WriteJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

// WriteError writes a JSON error response with the given status and message.
func WriteError(w http.ResponseWriter, status int, msg string) {
	WriteJSON(w, status, map[string]any{"error": msg})
}

// WriteSSE writes a single Server-Sent Event with the given JSON payload.
func WriteSSE(w http.ResponseWriter, payload map[string]any) {
	b, _ := json.Marshal(payload)
	_, _ = w.Write([]byte("data: " + string(b) + "\n\n"))
}
