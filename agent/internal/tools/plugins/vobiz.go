package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

const vobizAPI = "https://api.vobiz.ai/api/v1"

type MakePhoneCallTool struct{}
type GetUserPhoneNumberTool struct{}

func (t *MakePhoneCallTool) Definition() tools.Definition {
	return tools.Definition{Name: "make_phone_call", Description: "Make outbound call via Vobiz.", StatusText: "Initiating phone call...", Parameters: []tools.Param{{Name: "to_number", Type: "string", Required: false}, {Name: "greeting", Type: "string", Required: false}, {Name: "reason", Type: "string", Required: false}}}
}

func (t *MakePhoneCallTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	authID, err := requireString(cfg, "auth_id")
	if err != nil {
		return tools.Fail("Vobiz is not configured: missing auth_id", nil)
	}
	authToken, err := requireString(cfg, "auth_token")
	if err != nil {
		return tools.Fail("Vobiz is not configured: missing auth_token", nil)
	}
	fromNumber, err := requireString(cfg, "from_number")
	if err != nil {
		return tools.Fail("Vobiz is not configured: missing from_number", nil)
	}
	toNumber, _ := call.Args["to_number"].(string)
	if strings.TrimSpace(toNumber) == "" {
		toNumber = strings.TrimSpace(cfg["user_phone_number"])
	}
	if toNumber == "" {
		return tools.Fail("No destination number provided and user_phone_number is not configured", nil)
	}
	baseURL := strings.TrimSpace(cfg["base_url"])
	if baseURL == "" {
		baseURL = strings.TrimSpace(cfg["public_base_url"])
	}
	if baseURL == "" {
		return tools.Fail("Missing base_url/public_base_url in vobiz config", nil)
	}
	greeting, _ := call.Args["greeting"].(string)
	answerURL := strings.TrimRight(baseURL, "/") + "/api/plugins/vobiz/answer"
	if strings.TrimSpace(greeting) != "" {
		answerURL += "?greeting=" + url.QueryEscape(greeting)
	}
	payload := map[string]any{
		"from":          strings.TrimPrefix(fromNumber, "+"),
		"to":            strings.TrimPrefix(toNumber, "+"),
		"answer_url":    answerURL,
		"answer_method": "POST",
		"caller_name":   "Aether",
	}
	b, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, vobizAPI+"/Account/"+url.PathEscape(authID)+"/Call/", strings.NewReader(string(b)))
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Auth-ID", authID)
	req.Header.Set("X-Auth-Token", authToken)
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return tools.Fail("Vobiz API request failed: "+err.Error(), nil)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		return tools.Fail(fmt.Sprintf("Vobiz API request failed with status %d", resp.StatusCode), nil)
	}
	var obj map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&obj)
	return tools.Success("Phone call initiated.", map[string]any{"call_uuid": obj["call_uuid"], "to_number": toNumber})
}

func (t *GetUserPhoneNumberTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_user_phone_number", Description: "Get configured user phone number.", StatusText: "Checking phone number..."}
}

func (t *GetUserPhoneNumberTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	if call.Ctx.PluginState == nil {
		return tools.Fail("plugin state is unavailable", nil)
	}
	cfg, err := call.Ctx.PluginState.Config(ctx)
	if err != nil {
		return tools.Fail("failed to read plugin config: "+err.Error(), nil)
	}
	phone := strings.TrimSpace(cfg["user_phone_number"])
	if phone == "" {
		return tools.Success("No user phone number configured.", nil)
	}
	return tools.Success("Configured user phone number: "+phone, map[string]any{"phone_number": phone})
}

var (
	_ tools.Tool = (*MakePhoneCallTool)(nil)
	_ tools.Tool = (*GetUserPhoneNumberTool)(nil)
)
