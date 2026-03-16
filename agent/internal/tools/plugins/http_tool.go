package plugins

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	coreplugins "github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// HTTPTool is a generic tool implementation driven entirely by a manifest definition.
// It reads the tool's HTTP config, injects auth, makes the API call, and returns the result.
type HTTPTool struct {
	pluginName string
	manifest   coreplugins.PluginManifest
	toolDef    coreplugins.ManifestTool
}

func NewHTTPTool(pluginName string, manifest coreplugins.PluginManifest, toolDef coreplugins.ManifestTool) *HTTPTool {
	return &HTTPTool{pluginName: pluginName, manifest: manifest, toolDef: toolDef}
}

func (t *HTTPTool) Definition() tools.Definition {
	params := make([]tools.Param, 0, len(t.toolDef.Parameters))
	for _, p := range t.toolDef.Parameters {
		params = append(params, tools.Param{
			Name:        p.Name,
			Type:        p.Type,
			Description: p.Description,
			Required:    p.Required,
			Default:     p.Default,
		})
	}
	return tools.Definition{
		Name:        t.toolDef.Name,
		Description: t.toolDef.Description,
		StatusText:  t.toolDef.StatusText,
		Parameters:  params,
	}
}

func (t *HTTPTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	// 1. Build URL from base + path with template substitution.
	baseURL := strings.TrimRight(t.manifest.API.BaseURL, "/")
	path := t.toolDef.HTTP.Path
	method := strings.ToUpper(t.toolDef.HTTP.Method)

	// Collect params by mapping target.
	queryParams := url.Values{}
	bodyFields := map[string]any{}

	// Add static query params from manifest.
	for k, v := range t.toolDef.HTTP.Query {
		queryParams.Set(k, v)
	}

	// Add static body fields from manifest.
	for k, v := range t.toolDef.HTTP.Body {
		bodyFields[k] = v
	}

	// Process tool call arguments.
	for _, paramDef := range t.toolDef.Parameters {
		val, ok := call.Args[paramDef.Name]
		if !ok {
			if paramDef.Default != nil {
				val = paramDef.Default
			} else {
				continue
			}
		}

		mapTo := strings.TrimSpace(paramDef.MapTo)
		if mapTo == "" {
			// Auto-detect: if the param name appears in the path template, map to path.
			if strings.Contains(path, "{{"+paramDef.Name+"}}") {
				mapTo = "path." + paramDef.Name
			} else {
				// Default: body for POST/PUT/PATCH, query for GET/DELETE.
				if method == "POST" || method == "PUT" || method == "PATCH" {
					mapTo = "body." + paramDef.Name
				} else {
					mapTo = "query." + paramDef.Name
				}
			}
		}

		parts := strings.SplitN(mapTo, ".", 2)
		if len(parts) != 2 {
			continue
		}
		target, field := parts[0], parts[1]
		switch target {
		case "query":
			queryParams.Set(field, fmt.Sprintf("%v", val))
		case "path":
			path = strings.ReplaceAll(path, "{{"+field+"}}", fmt.Sprintf("%v", val))
		case "body":
			bodyFields[field] = val
		}
	}

	normalizeToolRequest(t.pluginName, t.toolDef.Name, call.Args, queryParams, bodyFields, &method, &path)

	// Apply transform if specified (for complex request building like MIME email).
	if t.toolDef.Transform != "" {
		transformed, err := applyTransform(t.toolDef.Transform, call.Args, bodyFields)
		if err != nil {
			return tools.Fail("Transform failed: "+err.Error(), nil)
		}
		bodyFields = transformed
	}

	// Build full URL.
	fullURL := baseURL + path
	if len(queryParams) > 0 {
		fullURL += "?" + queryParams.Encode()
	}

	// Build request body.
	var reqBody io.Reader
	if len(bodyFields) > 0 && (method == "POST" || method == "PUT" || method == "PATCH") {
		b, _ := json.Marshal(bodyFields)
		reqBody = strings.NewReader(string(b))
	} else {
		reqBody = strings.NewReader("")
	}

	// 2. Create HTTP request.
	req, err := http.NewRequestWithContext(ctx, method, fullURL, reqBody)
	if err != nil {
		return tools.Fail("Failed to build request: "+err.Error(), nil)
	}

	// Set default headers.
	for k, v := range t.manifest.API.Headers {
		req.Header.Set(k, v)
	}
	// Set per-tool headers.
	for k, v := range t.toolDef.HTTP.Headers {
		req.Header.Set(k, v)
	}
	if reqBody != nil && (method == "POST" || method == "PUT" || method == "PATCH") {
		if req.Header.Get("Content-Type") == "" {
			req.Header.Set("Content-Type", "application/json")
		}
	}

	// 3. Inject auth.
	if err := t.injectAuth(ctx, call, req); err != nil {
		return tools.Fail(err.Error(), nil)
	}

	// 4. Execute request.
	timeout := 20 * time.Second
	if t.manifest.API.Timeout > 0 {
		timeout = time.Duration(t.manifest.API.Timeout) * time.Second
	}
	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return tools.Fail("API request failed: "+err.Error(), nil)
	}
	defer resp.Body.Close()

	// 5. Handle 401 — auto-refresh for OAuth and retry once.
	if resp.StatusCode == http.StatusUnauthorized && t.manifest.Auth.Type == "oauth2" && t.manifest.Auth.AutoRefresh {
		resp.Body.Close()
		if t.refreshOAuthToken(ctx, call) {
			// Rebuild request for retry.
			if reqBody != nil && len(bodyFields) > 0 {
				b, _ := json.Marshal(bodyFields)
				reqBody = strings.NewReader(string(b))
			} else {
				reqBody = strings.NewReader("")
			}
			retryReq, err := http.NewRequestWithContext(ctx, method, fullURL, reqBody)
			if err != nil {
				return tools.Fail("Failed to build retry request: "+err.Error(), nil)
			}
			for k, v := range t.manifest.API.Headers {
				retryReq.Header.Set(k, v)
			}
			for k, v := range t.toolDef.HTTP.Headers {
				retryReq.Header.Set(k, v)
			}
			if method == "POST" || method == "PUT" || method == "PATCH" {
				if retryReq.Header.Get("Content-Type") == "" {
					retryReq.Header.Set("Content-Type", "application/json")
				}
			}
			if err := t.injectAuth(ctx, call, retryReq); err != nil {
				return tools.Fail(err.Error(), nil)
			}
			resp, err = (&http.Client{Timeout: timeout}).Do(retryReq)
			if err != nil {
				return tools.Fail("API request failed after token refresh: "+err.Error(), nil)
			}
			defer resp.Body.Close()
		} else {
			return tools.Fail(t.manifest.DisplayName+" token refresh failed; please reconnect the plugin.", nil)
		}
	}

	// 6. Handle error responses.
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		msg := buildHTTPStatusErrorMessage(t.manifest.DisplayName, resp.StatusCode, body)
		return tools.Fail(msg, nil)
	}

	// 7. Handle no-content responses.
	if resp.StatusCode == http.StatusNoContent {
		successMsg := t.toolDef.Response.SuccessMsg
		if successMsg == "" {
			successMsg = "Done."
		}
		return tools.Success(successMsg, nil)
	}

	// 8. Read and optionally extract response.
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return tools.Fail("Failed to read response: "+err.Error(), nil)
	}

	output := string(respBody)
	if extract := strings.TrimSpace(t.toolDef.Response.Extract); extract != "" {
		var obj map[string]any
		if err := json.Unmarshal(respBody, &obj); err == nil {
			if extracted, ok := obj[extract]; ok {
				if t.pluginName == "google-calendar" && t.toolDef.Name == "upcoming_events" {
					extracted = filterCalendarEventsByWindow(extracted, queryParams.Get("timeMin"), queryParams.Get("timeMax"))
				}
				b, _ := json.Marshal(extracted)
				output = string(b)
			}
		}
	}

	successMsg := t.toolDef.Response.SuccessMsg
	if successMsg != "" {
		// Return success message + extracted data.
		return tools.Success(successMsg+"\n"+output, nil)
	}
	return tools.Success(output, nil)
}

// injectAuth adds the appropriate auth header to the request based on manifest auth type.
func (t *HTTPTool) injectAuth(ctx context.Context, call tools.Call, req *http.Request) error {
	switch t.manifest.Auth.Type {
	case "none":
		return nil

	case "api_key":
		cfg, err := pluginConfig(ctx, call)
		if err != nil {
			return fmt.Errorf("%s is not configured: %v", t.manifest.DisplayName, err)
		}
		configKey := t.manifest.Auth.ConfigKey
		if configKey == "" {
			configKey = "api_key"
		}
		key, err := requireString(ctx, call, cfg, configKey)
		if err != nil {
			return fmt.Errorf("%s is not connected: missing %s", t.manifest.DisplayName, configKey)
		}
		headerName := t.manifest.Auth.HeaderName
		if headerName == "" {
			headerName = "X-Api-Key"
		}
		req.Header.Set(headerName, key)
		return nil

	case "bearer":
		cfg, err := pluginConfig(ctx, call)
		if err != nil {
			return fmt.Errorf("%s is not configured: %v", t.manifest.DisplayName, err)
		}
		configKey := t.manifest.Auth.ConfigKey
		if configKey == "" {
			configKey = "access_token"
		}
		token, err := requireString(ctx, call, cfg, configKey)
		if err != nil {
			return fmt.Errorf("%s is not connected: missing %s", t.manifest.DisplayName, configKey)
		}
		req.Header.Set("Authorization", "Bearer "+token)
		return nil

	case "oauth2":
		cfg, err := pluginConfig(ctx, call)
		if err != nil {
			return fmt.Errorf("%s is not configured: %v", t.manifest.DisplayName, err)
		}
		token, err := requireToken(ctx, call, cfg)
		if err != nil {
			return fmt.Errorf("%s is not connected: %v", t.manifest.DisplayName, err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
		return nil

	default:
		return nil
	}
}

// refreshOAuthToken performs an OAuth2 token refresh using the manifest's token URL.
func (t *HTTPTool) refreshOAuthToken(ctx context.Context, call tools.Call) bool {
	tokenURL := t.manifest.Auth.TokenURL
	if tokenURL == "" {
		return false
	}
	result := refreshOAuthAccessToken(ctx, call, tokenURL, t.manifest.Auth.UseBasicAuth)
	if result.Error {
		if call.Ctx.PluginState != nil {
			cfg, err := call.Ctx.PluginState.Config(ctx)
			if err == nil {
				cfg["needs_reconnect"] = "true"
				_ = call.Ctx.PluginState.SetConfig(ctx, cfg)
			}
		}
		return false
	}
	return true
}

// applyTransform applies a named transform to build complex request bodies.
func applyTransform(name string, args map[string]any, body map[string]any) (map[string]any, error) {
	switch name {
	case "mime_message":
		return transformMIMEMessage(args, body)
	case "mime_reply":
		return transformMIMEReply(args, body)
	case "mime_draft":
		return transformMIMEDraft(args, body)
	case "calendar_event":
		return transformCalendarEvent(args, body)
	case "label_modify":
		return transformLabelModify(args, body)
	default:
		return body, nil
	}
}

func transformMIMEMessage(args map[string]any, _ map[string]any) (map[string]any, error) {
	to, _ := args["to"].(string)
	subject, _ := args["subject"].(string)
	body, _ := args["body"].(string)
	cc, _ := args["cc"].(string)
	bcc, _ := args["bcc"].(string)
	headers := []string{"To: " + to, "Subject: " + subject}
	if strings.TrimSpace(cc) != "" {
		headers = append(headers, "Cc: "+cc)
	}
	if strings.TrimSpace(bcc) != "" {
		headers = append(headers, "Bcc: "+bcc)
	}
	raw := base64.RawURLEncoding.EncodeToString([]byte(strings.Join(headers, "\r\n") + "\r\n\r\n" + body))
	return map[string]any{"raw": raw}, nil
}

func transformMIMEReply(args map[string]any, _ map[string]any) (map[string]any, error) {
	to, _ := args["to"].(string)
	subject, _ := args["subject"].(string)
	body, _ := args["body"].(string)
	threadID, _ := args["thread_id"].(string)
	if strings.TrimSpace(subject) == "" {
		subject = "Re: (no subject)"
	}
	raw := base64.RawURLEncoding.EncodeToString([]byte("To: " + to + "\r\nSubject: " + subject + "\r\n\r\n" + body))
	return map[string]any{"raw": raw, "threadId": threadID}, nil
}

func transformMIMEDraft(args map[string]any, _ map[string]any) (map[string]any, error) {
	msg, err := transformMIMEMessage(args, nil)
	if err != nil {
		return nil, err
	}
	return map[string]any{"message": msg}, nil
}

func transformCalendarEvent(args map[string]any, _ map[string]any) (map[string]any, error) {
	summary, _ := args["summary"].(string)
	startTime, _ := args["start_time"].(string)
	endTime, _ := args["end_time"].(string)
	desc, _ := args["description"].(string)
	location, _ := args["location"].(string)
	event := map[string]any{
		"summary": summary,
		"start":   map[string]any{"dateTime": startTime},
		"end":     map[string]any{"dateTime": endTime},
	}
	if strings.TrimSpace(desc) != "" {
		event["description"] = desc
	}
	if strings.TrimSpace(location) != "" {
		event["location"] = location
	}
	return event, nil
}

func transformLabelModify(args map[string]any, _ map[string]any) (map[string]any, error) {
	action, _ := args["_label_action"].(string) // "addLabelIds" or "removeLabelIds"
	labelIDs, _ := args["_label_ids"].([]string)
	if action == "" || len(labelIDs) == 0 {
		return nil, fmt.Errorf("label_modify requires _label_action and _label_ids")
	}
	return map[string]any{action: labelIDs}, nil
}

func applyUpcomingEventsWindow(args map[string]any, queryParams url.Values) {
	days := intArg(args, "days", 7)
	if days <= 0 {
		days = 7
	}
	if days > 90 {
		days = 90
	}

	start := time.Now().UTC()
	end := start.AddDate(0, 0, days)

	queryParams.Del("days")
	queryParams.Set("singleEvents", "true")
	queryParams.Set("orderBy", "startTime")
	queryParams.Set("timeMin", start.Format(time.RFC3339))
	queryParams.Set("timeMax", end.Format(time.RFC3339))
}

type requestNormalizerContext struct {
	args        map[string]any
	queryParams url.Values
	bodyFields  map[string]any
	method      *string
	path        *string
}

type requestNormalizer func(ctx requestNormalizerContext)

func normalizerKey(pluginName, toolName string) string {
	return strings.TrimSpace(strings.ToLower(pluginName)) + ":" + strings.TrimSpace(strings.ToLower(toolName))
}

var toolRequestNormalizers = map[string]requestNormalizer{
	normalizerKey("google-calendar", "upcoming_events"): func(ctx requestNormalizerContext) {
		applyUpcomingEventsWindow(ctx.args, ctx.queryParams)
	},
	normalizerKey("google-drive", "list_drive_files"): func(ctx requestNormalizerContext) {
		applyDriveFolderFilter(ctx.args, ctx.queryParams)
	},
	normalizerKey("google-drive", "search_drive"): func(ctx requestNormalizerContext) {
		normalizeDriveSearchQuery(ctx.args, ctx.queryParams)
	},
	normalizerKey("google-drive", "create_folder"): func(ctx requestNormalizerContext) {
		normalizeDriveCreateFolder(ctx.args, ctx.bodyFields)
	},
	normalizerKey("spotify", "play_pause"): func(ctx requestNormalizerContext) {
		normalizeSpotifyPlayPause(ctx.args, ctx.bodyFields, ctx.method, ctx.path)
	},
}

func normalizeToolRequest(pluginName, toolName string, args map[string]any, queryParams url.Values, bodyFields map[string]any, method, path *string) {
	if fn, ok := toolRequestNormalizers[normalizerKey(pluginName, toolName)]; ok {
		fn(requestNormalizerContext{
			args:        args,
			queryParams: queryParams,
			bodyFields:  bodyFields,
			method:      method,
			path:        path,
		})
	}
}

func applyDriveFolderFilter(args map[string]any, queryParams url.Values) {
	folderID := strings.TrimSpace(stringArg(args, "folder_id", ""))
	if folderID == "" {
		folderID = strings.TrimSpace(queryParams.Get("folder_id"))
	}
	if folderID == "" {
		folderID = "root"
	}
	queryParams.Del("folder_id")
	queryParams.Set("q", fmt.Sprintf("'%s' in parents and trashed = false", escapeSingleQuote(folderID)))
}

func normalizeDriveSearchQuery(args map[string]any, queryParams url.Values) {
	raw := strings.TrimSpace(stringArg(args, "query", ""))
	if raw == "" {
		raw = strings.TrimSpace(queryParams.Get("q"))
	}
	if raw == "" {
		return
	}
	if looksLikeDriveQuery(raw) {
		queryParams.Set("q", ensureDriveNotTrashedClause(raw))
		return
	}

	terms := strings.Fields(raw)
	clauses := make([]string, 0, len(terms)+1)
	for _, term := range terms {
		term = strings.TrimSpace(term)
		if term == "" {
			continue
		}
		clauses = append(clauses, fmt.Sprintf("name contains '%s'", escapeSingleQuote(term)))
		if len(clauses) >= 8 {
			break
		}
	}
	if len(clauses) == 0 {
		clauses = append(clauses, fmt.Sprintf("name contains '%s'", escapeSingleQuote(raw)))
	}
	clauses = append(clauses, "trashed = false")
	queryParams.Set("q", strings.Join(clauses, " and "))
}

func normalizeDriveCreateFolder(args map[string]any, bodyFields map[string]any) {
	if strings.TrimSpace(stringArg(args, "name", "")) == "" {
		if strings.TrimSpace(fmt.Sprintf("%v", bodyFields["name"])) == "" {
			return
		}
	}

	parentID := strings.TrimSpace(stringArg(args, "parent_id", ""))
	if parentID == "" {
		if raw, ok := bodyFields["parent_id"]; ok {
			parentID = strings.TrimSpace(fmt.Sprintf("%v", raw))
		}
	}
	if parentID == "" {
		parentID = "root"
	}
	bodyFields["mimeType"] = "application/vnd.google-apps.folder"
	bodyFields["parents"] = []string{parentID}
	delete(bodyFields, "parent_id")
}

func normalizeSpotifyPlayPause(args map[string]any, bodyFields map[string]any, method, path *string) {
	action := strings.ToLower(strings.TrimSpace(stringArg(args, "action", "")))
	if action == "" {
		action = strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", bodyFields["action"])))
	}
	if action == "pause" {
		*method = "PUT"
		*path = "/me/player/pause"
		delete(bodyFields, "action")
		return
	}
	// For play and toggle, default to /play.
	*method = "PUT"
	*path = "/me/player/play"
	delete(bodyFields, "action")
}

func intArg(args map[string]any, key string, fallback int) int {
	if args == nil {
		return fallback
	}
	v, ok := args[key]
	if !ok || v == nil {
		return fallback
	}
	switch n := v.(type) {
	case int:
		return n
	case int32:
		return int(n)
	case int64:
		return int(n)
	case float32:
		return int(n)
	case float64:
		return int(n)
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return int(i)
		}
	case string:
		n = strings.TrimSpace(n)
		if n == "" {
			return fallback
		}
		if i, err := json.Number(n).Int64(); err == nil {
			return int(i)
		}
	}
	return fallback
}

func stringArg(args map[string]any, key, fallback string) string {
	if args == nil {
		return fallback
	}
	v, ok := args[key]
	if !ok || v == nil {
		return fallback
	}
	if s, ok := v.(string); ok {
		trimmed := strings.TrimSpace(s)
		if trimmed == "" {
			return fallback
		}
		return trimmed
	}
	trimmed := strings.TrimSpace(fmt.Sprintf("%v", v))
	if trimmed == "" {
		return fallback
	}
	return trimmed
}

func escapeSingleQuote(value string) string {
	return strings.ReplaceAll(value, "'", "\\'")
}

func looksLikeDriveQuery(value string) bool {
	v := strings.ToLower(strings.TrimSpace(value))
	if v == "" {
		return false
	}
	patterns := []string{" contains ", " and ", " or ", "=", " in parents", " mimetype", " modifiedtime", "name ", "trashed"}
	for _, pattern := range patterns {
		if strings.Contains(v, pattern) {
			return true
		}
	}
	return strings.Contains(v, "'")
}

func ensureDriveNotTrashedClause(query string) string {
	q := strings.TrimSpace(query)
	if q == "" {
		return "trashed = false"
	}
	if strings.Contains(strings.ToLower(q), "trashed") {
		return q
	}
	return q + " and trashed = false"
}

func filterCalendarEventsByWindow(extracted any, timeMinRaw, timeMaxRaw string) any {
	items, ok := extracted.([]any)
	if !ok {
		return extracted
	}
	if len(items) == 0 {
		return items
	}
	timeMin, errMin := time.Parse(time.RFC3339, strings.TrimSpace(timeMinRaw))
	timeMax, errMax := time.Parse(time.RFC3339, strings.TrimSpace(timeMaxRaw))
	if errMin != nil || errMax != nil {
		return items
	}

	filtered := make([]any, 0, len(items))
	for _, raw := range items {
		event, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		start := eventStartTime(event)
		if start.IsZero() {
			continue
		}
		if (start.Equal(timeMin) || start.After(timeMin)) && start.Before(timeMax) {
			filtered = append(filtered, raw)
		}
	}
	return filtered
}

func eventStartTime(event map[string]any) time.Time {
	startRaw, ok := event["start"].(map[string]any)
	if !ok {
		return time.Time{}
	}
	if dt, ok := startRaw["dateTime"].(string); ok {
		if ts, err := time.Parse(time.RFC3339, strings.TrimSpace(dt)); err == nil {
			return ts
		}
	}
	if d, ok := startRaw["date"].(string); ok {
		if ts, err := time.Parse("2006-01-02", strings.TrimSpace(d)); err == nil {
			return ts
		}
	}
	return time.Time{}
}

func buildHTTPStatusErrorMessage(displayName string, statusCode int, body []byte) string {
	statusText := strings.TrimSpace(strings.ToLower(http.StatusText(statusCode)))
	base := fmt.Sprintf("%s API returned status %d", displayName, statusCode)

	switch {
	case statusCode == http.StatusBadRequest:
		base = fmt.Sprintf("%s API bad request (400)", displayName)
	case statusCode == http.StatusUnauthorized:
		base = fmt.Sprintf("%s API unauthorized (401)", displayName)
	case statusCode == http.StatusForbidden:
		base = fmt.Sprintf("%s API forbidden (403)", displayName)
	case statusCode == http.StatusTooManyRequests:
		base = fmt.Sprintf("%s API rate limit exceeded (429)", displayName)
	case statusCode == http.StatusServiceUnavailable || statusCode == http.StatusBadGateway || statusCode == http.StatusGatewayTimeout:
		base = fmt.Sprintf("%s API temporarily unavailable (%d)", displayName, statusCode)
	case statusCode >= 500:
		base = fmt.Sprintf("%s API server error (%d)", displayName, statusCode)
	case statusText != "":
		base = fmt.Sprintf("%s API %s (%d)", displayName, statusText, statusCode)
	}

	bodyText := strings.TrimSpace(string(body))
	if bodyText != "" && len(bodyText) < 500 {
		base += ": " + bodyText
	}
	return base
}

var _ tools.Tool = (*HTTPTool)(nil)
