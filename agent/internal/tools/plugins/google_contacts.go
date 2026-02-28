package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

const peopleAPI = "https://people.googleapis.com/v1"

type SearchContactsTool struct{}
type GetContactTool struct{}
type RefreshGoogleContactsTokenTool struct{}

func (t *SearchContactsTool) Definition() tools.Definition {
	return tools.Definition{Name: "search_contacts", Description: "Search Google Contacts by name/email/phone.", StatusText: "Searching contacts...", Parameters: []tools.Param{{Name: "query", Type: "string", Required: true}, {Name: "max_results", Type: "integer", Required: false, Default: 10}}}
}

func (t *SearchContactsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	query, _ := call.Args["query"].(string)
	max, _ := asInt(call.Args["max_results"])
	if max <= 0 {
		max = 10
	}
	v := url.Values{}
	v.Set("query", query)
	v.Set("readMask", "names,emailAddresses,phoneNumbers,organizations")
	v.Set("pageSize", fmt.Sprintf("%d", max))
	obj, err := contactsRequest(ctx, call, http.MethodGet, peopleAPI+"/people:searchContacts?"+v.Encode())
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["results"])
	return tools.Success(string(b), map[string]any{"query": query})
}

func (t *GetContactTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_contact", Description: "Get Google Contact by resource name.", StatusText: "Loading contact...", Parameters: []tools.Param{{Name: "resource_name", Type: "string", Required: true}}}
}

func (t *GetContactTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	resourceName, _ := call.Args["resource_name"].(string)
	v := url.Values{}
	v.Set("personFields", "names,emailAddresses,phoneNumbers,organizations,addresses,birthdays,biographies")
	obj, err := contactsRequest(ctx, call, http.MethodGet, peopleAPI+"/"+url.PathEscape(resourceName)+"?"+v.Encode())
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj)
	return tools.Success(string(b), map[string]any{"resource_name": resourceName})
}

func (t *RefreshGoogleContactsTokenTool) Definition() tools.Definition {
	return tools.Definition{Name: "refresh_google_contacts_token", Description: "Refresh Google Contacts OAuth access token.", StatusText: "Refreshing Contacts token..."}
}

func (t *RefreshGoogleContactsTokenTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return refreshOAuthAccessToken(ctx, call, "https://oauth2.googleapis.com/token", false)
}

func contactsRequest(ctx context.Context, call tools.Call, method, reqURL string) (map[string]any, error) {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return nil, err
	}
	token, err := requireToken(ctx, call, cfg)
	if err != nil {
		return nil, fmt.Errorf("Google Contacts is not connected: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, method, reqURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Google Contacts API request failed with status %d", resp.StatusCode)
	}
	var obj map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&obj); err != nil {
		return nil, err
	}
	return obj, nil
}

var (
	_ tools.Tool = (*SearchContactsTool)(nil)
	_ tools.Tool = (*GetContactTool)(nil)
	_ tools.Tool = (*RefreshGoogleContactsTokenTool)(nil)
)
