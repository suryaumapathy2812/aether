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

const calendarAPI = "https://www.googleapis.com/calendar/v3"

type UpcomingEventsTool struct{}
type SearchEventsTool struct{}
type CreateEventTool struct{}
type GetEventTool struct{}
type RefreshGoogleCalendarTokenTool struct{}

func (t *UpcomingEventsTool) Definition() tools.Definition {
	return tools.Definition{Name: "upcoming_events", Description: "Get upcoming Google Calendar events.", StatusText: "Loading calendar events...", Parameters: []tools.Param{{Name: "days", Type: "integer", Required: false, Default: 7}, {Name: "max_results", Type: "integer", Required: false, Default: 10}}}
}

func (t *UpcomingEventsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	days, _ := asInt(call.Args["days"])
	max, _ := asInt(call.Args["max_results"])
	if days <= 0 {
		days = 7
	}
	if max <= 0 {
		max = 10
	}
	now := time.Now().UTC()
	v := url.Values{}
	v.Set("timeMin", now.Format(time.RFC3339))
	v.Set("timeMax", now.Add(time.Duration(days)*24*time.Hour).Format(time.RFC3339))
	v.Set("maxResults", fmt.Sprintf("%d", max))
	v.Set("singleEvents", "true")
	v.Set("orderBy", "startTime")
	obj, err := calendarRequest(ctx, call, http.MethodGet, calendarAPI+"/calendars/primary/events?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["items"])
	return tools.Success(string(b), map[string]any{"days": days, "max_results": max})
}

func (t *SearchEventsTool) Definition() tools.Definition {
	return tools.Definition{Name: "search_events", Description: "Search Google Calendar events by keyword.", StatusText: "Searching calendar...", Parameters: []tools.Param{{Name: "query", Type: "string", Required: true}, {Name: "max_results", Type: "integer", Required: false, Default: 10}}}
}

func (t *SearchEventsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	query, _ := call.Args["query"].(string)
	max, _ := asInt(call.Args["max_results"])
	if max <= 0 {
		max = 10
	}
	v := url.Values{}
	v.Set("q", query)
	v.Set("timeMin", time.Now().UTC().Format(time.RFC3339))
	v.Set("maxResults", fmt.Sprintf("%d", max))
	v.Set("singleEvents", "true")
	v.Set("orderBy", "startTime")
	obj, err := calendarRequest(ctx, call, http.MethodGet, calendarAPI+"/calendars/primary/events?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["items"])
	return tools.Success(string(b), map[string]any{"query": query})
}

func (t *CreateEventTool) Definition() tools.Definition {
	return tools.Definition{Name: "create_event", Description: "Create a Google Calendar event.", StatusText: "Creating event...", Parameters: []tools.Param{{Name: "summary", Type: "string", Required: true}, {Name: "start_time", Type: "string", Required: true}, {Name: "end_time", Type: "string", Required: true}, {Name: "description", Type: "string", Required: false, Default: ""}, {Name: "location", Type: "string", Required: false, Default: ""}}}
}

func (t *CreateEventTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	summary, _ := call.Args["summary"].(string)
	startTime, _ := call.Args["start_time"].(string)
	endTime, _ := call.Args["end_time"].(string)
	desc, _ := call.Args["description"].(string)
	location, _ := call.Args["location"].(string)
	body := map[string]any{"summary": summary, "start": map[string]any{"dateTime": startTime}, "end": map[string]any{"dateTime": endTime}}
	if strings.TrimSpace(desc) != "" {
		body["description"] = desc
	}
	if strings.TrimSpace(location) != "" {
		body["location"] = location
	}
	obj, err := calendarRequest(ctx, call, http.MethodPost, calendarAPI+"/calendars/primary/events", body)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Event created.", map[string]any{"event_id": obj["id"], "link": obj["htmlLink"]})
}

func (t *GetEventTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_event", Description: "Get a Google Calendar event by id.", StatusText: "Loading event...", Parameters: []tools.Param{{Name: "event_id", Type: "string", Required: true}}}
}

func (t *GetEventTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	eventID, _ := call.Args["event_id"].(string)
	obj, err := calendarRequest(ctx, call, http.MethodGet, calendarAPI+"/calendars/primary/events/"+url.PathEscape(eventID), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj)
	return tools.Success(string(b), map[string]any{"event_id": eventID})
}

func (t *RefreshGoogleCalendarTokenTool) Definition() tools.Definition {
	return tools.Definition{Name: "refresh_google_calendar_token", Description: "Refresh Google Calendar OAuth access token.", StatusText: "Refreshing Calendar token..."}
}

func (t *RefreshGoogleCalendarTokenTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return refreshOAuthAccessToken(ctx, call, "https://oauth2.googleapis.com/token", false)
}

type SetupCalendarWatchTool struct{}

func (t *SetupCalendarWatchTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "setup_calendar_watch",
		Description: "Set up Google Calendar push notifications via Google Pub/Sub.",
		StatusText:  "Setting up Calendar watch...",
		Parameters: []tools.Param{
			{
				Name:        "pubsub_topic",
				Type:        "string",
				Required:    false,
				Description: "Google Cloud Pub/Sub topic name (projects/*/topics/*).",
			},
		},
	}
}

func (t *SetupCalendarWatchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail("Failed to get plugin config: "+err.Error(), nil)
	}

	token, err := requireToken(ctx, call, cfg)
	if err != nil {
		return tools.Fail("Google Calendar is not connected: "+err.Error(), nil)
	}

	topicName, _ := call.Args["pubsub_topic"].(string)
	if topicName == "" {
		topicName = cfg["pubsub_topic"]
	}
	if topicName == "" {
		topicName = cfg["calendar_topic"]
	}
	if topicName == "" {
		return tools.Fail("Missing pubsub_topic. Provide as parameter or configure in plugin settings.", nil)
	}

	publicBaseURL := cfg["public_base_url"]
	if publicBaseURL == "" {
		publicBaseURL = cfg["base_url"]
	}
	if publicBaseURL == "" {
		return tools.Fail("Missing public_base_url in plugin config.", nil)
	}

	webhookURL := strings.TrimRight(publicBaseURL, "/") + "/api/hooks/pubsub/google-calendar"

	watchRequest := map[string]any{
		"id":      "watch-" + time.Now().Format("20060102150405"),
		"type":    "web_hook",
		"address": webhookURL,
	}

	b, err := json.Marshal(watchRequest)
	if err != nil {
		return tools.Fail("Failed to create request: "+err.Error(), nil)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, calendarAPI+"/calendars/primary/events/watch", strings.NewReader(string(b)))
	if err != nil {
		return tools.Fail("Failed to create request: "+err.Error(), nil)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return tools.Fail("Calendar API request failed: "+err.Error(), nil)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return tools.Fail("Failed to setup Calendar watch: status "+fmt.Sprintf("%d", resp.StatusCode), nil)
	}

	cfg["pubsub_topic"] = topicName
	cfg["calendar_topic"] = topicName
	_ = call.Ctx.PluginState.SetConfig(ctx, cfg)

	return tools.Success("Calendar watch set up. Webhook URL: "+webhookURL, map[string]any{
		"webhook_url":  webhookURL,
		"pubsub_topic": topicName,
	})
}

func calendarRequest(ctx context.Context, call tools.Call, method, reqURL string, body any) (map[string]any, error) {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return nil, err
	}
	token, err := requireToken(ctx, call, cfg)
	if err != nil {
		return nil, fmt.Errorf("Google Calendar is not connected: %w", err)
	}
	var reqBody *strings.Reader
	if body == nil {
		reqBody = strings.NewReader("")
	} else {
		b, _ := json.Marshal(body)
		reqBody = strings.NewReader(string(b))
	}
	req, err := http.NewRequestWithContext(ctx, method, reqURL, reqBody)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Google Calendar API request failed with status %d", resp.StatusCode)
	}
	if resp.StatusCode == http.StatusNoContent {
		return map[string]any{}, nil
	}
	var obj map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&obj); err != nil {
		return nil, err
	}
	return obj, nil
}

var (
	_ tools.Tool = (*UpcomingEventsTool)(nil)
	_ tools.Tool = (*SearchEventsTool)(nil)
	_ tools.Tool = (*CreateEventTool)(nil)
	_ tools.Tool = (*GetEventTool)(nil)
	_ tools.Tool = (*RefreshGoogleCalendarTokenTool)(nil)
	_ tools.Tool = (*SetupCalendarWatchTool)(nil)
)
