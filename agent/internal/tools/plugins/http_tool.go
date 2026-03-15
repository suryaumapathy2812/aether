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
			// Default: body for POST/PUT/PATCH, query for GET/DELETE.
			method := strings.ToUpper(t.toolDef.HTTP.Method)
			if method == "POST" || method == "PUT" || method == "PATCH" {
				mapTo = "body." + paramDef.Name
			} else {
				mapTo = "query." + paramDef.Name
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
	method := strings.ToUpper(t.toolDef.HTTP.Method)
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
		msg := fmt.Sprintf("%s API returned status %d", t.manifest.DisplayName, resp.StatusCode)
		if len(body) > 0 && len(body) < 500 {
			msg += ": " + string(body)
		}
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

var _ tools.Tool = (*HTTPTool)(nil)
