package ws

import (
	"encoding/json"
	"net/http"
	"regexp"
	"strings"

	agentauth "github.com/suryaumapathy2812/core-ai/agent/internal/auth"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/httputil"
)

var modelNameRegex = regexp.MustCompile(`^[a-zA-Z0-9_\-./]+$`)

type PreferencesHandler struct {
	store     *db.Store
	validator *agentauth.Validator
}

func NewPreferencesHandler(store *db.Store, validator *agentauth.Validator) *PreferencesHandler {
	return &PreferencesHandler{store: store, validator: validator}
}

func (h *PreferencesHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/preferences/get", h.handleGet)
	mux.HandleFunc("/api/preferences/set", h.handleSet)
	mux.HandleFunc("/api/preferences/delete", h.handleDelete)
}

func (h *PreferencesHandler) handleGet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	userID, err := agentauth.ResolveDirectUserID(r, h.validator, r.URL.Query().Get("user_id"))
	key := strings.TrimSpace(r.URL.Query().Get("key"))
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	if key == "" {
		httputil.WriteError(w, http.StatusBadRequest, "key is required")
		return
	}

	value, err := h.store.GetUserPreference(r.Context(), userID, key)
	if err != nil {
		httputil.WriteError(w, http.StatusNotFound, "preference not found")
		return
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{"key": key, "value": value})
}

func (h *PreferencesHandler) handleSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req struct {
		UserID string `json:"user_id"`
		Key    string `json:"key"`
		Value  string `json:"value"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}

	userID, err := agentauth.ResolveDirectUserID(r, h.validator, req.UserID)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	key := strings.TrimSpace(req.Key)
	value := strings.TrimSpace(req.Value)

	if key == "" {
		httputil.WriteError(w, http.StatusBadRequest, "key is required")
		return
	}

	if key == "model" {
		if err := validateModelName(value); err != nil {
			httputil.WriteError(w, http.StatusBadRequest, err.Error())
			return
		}
	}

	err = h.store.SaveUserPreference(r.Context(), userID, key, value)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "saved", "key": key, "value": value})
}

func (h *PreferencesHandler) handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httputil.WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req struct {
		UserID string `json:"user_id"`
		Key    string `json:"key"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httputil.WriteError(w, http.StatusBadRequest, "invalid json body")
		return
	}

	userID, err := agentauth.ResolveDirectUserID(r, h.validator, req.UserID)
	if err != nil {
		httputil.WriteError(w, http.StatusUnauthorized, err.Error())
		return
	}
	key := strings.TrimSpace(req.Key)

	if key == "" {
		httputil.WriteError(w, http.StatusBadRequest, "key is required")
		return
	}

	err = h.store.DeleteUserPreference(r.Context(), userID, key)
	if err != nil {
		httputil.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}

	httputil.WriteJSON(w, http.StatusOK, map[string]any{"status": "deleted", "key": key})
}

func validateModelName(model string) error {
	if strings.TrimSpace(model) == "" {
		return nil
	}
	if len(model) > 200 {
		return ErrModelNameTooLong
	}
	if !modelNameRegex.MatchString(model) {
		return ErrInvalidModelName
	}
	return nil
}

var (
	ErrModelNameTooLong = &ValidationError{"model name must be under 200 characters"}
	ErrInvalidModelName = &ValidationError{"model name contains invalid characters"}
)

type ValidationError struct {
	Message string
}

func (e *ValidationError) Error() string {
	return e.Message
}
