package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

const spotifyAPI = "https://api.spotify.com/v1"

type NowPlayingTool struct{}
type PlayPauseTool struct{}
type SkipTrackTool struct{}
type SearchSpotifyTool struct{}
type QueueTrackTool struct{}
type RecentTracksTool struct{}
type RefreshSpotifyTokenTool struct{}

func (t *NowPlayingTool) Definition() tools.Definition {
	return tools.Definition{Name: "now_playing", Description: "Get currently playing Spotify track.", StatusText: "Checking Spotify..."}
}

func (t *NowPlayingTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	obj, code, err := spotifyRequest(ctx, call, http.MethodGet, spotifyAPI+"/me/player/currently-playing", nil)
	if err != nil {
		if code == http.StatusNoContent {
			return tools.Success("Nothing is currently playing.", nil)
		}
		return tools.Fail(err.Error(), nil)
	}
	item, _ := obj["item"].(map[string]any)
	if len(item) == 0 {
		return tools.Success("Nothing is currently playing.", nil)
	}
	track, _ := item["name"].(string)
	artists := []string{}
	if arr, ok := item["artists"].([]any); ok {
		for _, a := range arr {
			if m, ok := a.(map[string]any); ok {
				if n, ok := m["name"].(string); ok && n != "" {
					artists = append(artists, n)
				}
			}
		}
	}
	playing, _ := obj["is_playing"].(bool)
	state := "Paused"
	if playing {
		state = "Playing"
	}
	return tools.Success(fmt.Sprintf("%s: %s by %s", state, track, strings.Join(artists, ", ")), map[string]any{"track": track, "is_playing": playing})
}

func (t *PlayPauseTool) Definition() tools.Definition {
	return tools.Definition{Name: "play_pause", Description: "Play or pause Spotify playback.", StatusText: "Updating playback...", Parameters: []tools.Param{{Name: "action", Type: "string", Required: true, Enum: []string{"play", "pause"}}}}
}

func (t *PlayPauseTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	action, _ := call.Args["action"].(string)
	endpoint := "/me/player/pause"
	if action == "play" {
		endpoint = "/me/player/play"
	}
	_, _, err := spotifyRequest(ctx, call, http.MethodPut, spotifyAPI+endpoint, nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Playback updated.", map[string]any{"action": action})
}

func (t *SkipTrackTool) Definition() tools.Definition {
	return tools.Definition{Name: "skip_track", Description: "Skip track on Spotify.", StatusText: "Skipping track...", Parameters: []tools.Param{{Name: "direction", Type: "string", Required: false, Default: "next", Enum: []string{"next", "previous"}}}}
}

func (t *SkipTrackTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	direction, _ := call.Args["direction"].(string)
	if direction == "" {
		direction = "next"
	}
	_, _, err := spotifyRequest(ctx, call, http.MethodPost, spotifyAPI+"/me/player/"+direction, nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Track skipped.", map[string]any{"direction": direction})
}

func (t *SearchSpotifyTool) Definition() tools.Definition {
	return tools.Definition{Name: "search_spotify", Description: "Search Spotify for tracks, artists, or albums.", StatusText: "Searching Spotify...", Parameters: []tools.Param{{Name: "query", Type: "string", Required: true}, {Name: "search_type", Type: "string", Required: false, Default: "track", Enum: []string{"track", "artist", "album"}}, {Name: "limit", Type: "integer", Required: false, Default: 5}}}
}

func (t *SearchSpotifyTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	query, _ := call.Args["query"].(string)
	searchType, _ := call.Args["search_type"].(string)
	if searchType == "" {
		searchType = "track"
	}
	limit, _ := asInt(call.Args["limit"])
	if limit <= 0 {
		limit = 5
	}
	v := url.Values{}
	v.Set("q", query)
	v.Set("type", searchType)
	v.Set("limit", fmt.Sprintf("%d", limit))
	obj, _, err := spotifyRequest(ctx, call, http.MethodGet, spotifyAPI+"/search?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	key := searchType + "s"
	b, _ := json.Marshal(obj[key])
	return tools.Success(string(b), map[string]any{"query": query, "type": searchType})
}

func (t *QueueTrackTool) Definition() tools.Definition {
	return tools.Definition{Name: "queue_track", Description: "Queue a Spotify track URI.", StatusText: "Queueing track...", Parameters: []tools.Param{{Name: "uri", Type: "string", Required: true}}}
}

func (t *QueueTrackTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	uri, _ := call.Args["uri"].(string)
	v := url.Values{}
	v.Set("uri", uri)
	_, _, err := spotifyRequest(ctx, call, http.MethodPost, spotifyAPI+"/me/player/queue?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Track queued.", map[string]any{"uri": uri})
}

func (t *RecentTracksTool) Definition() tools.Definition {
	return tools.Definition{Name: "recent_tracks", Description: "Get recently played Spotify tracks.", StatusText: "Loading recent tracks...", Parameters: []tools.Param{{Name: "limit", Type: "integer", Required: false, Default: 10}}}
}

func (t *RecentTracksTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	limit, _ := asInt(call.Args["limit"])
	if limit <= 0 {
		limit = 10
	}
	v := url.Values{}
	v.Set("limit", fmt.Sprintf("%d", limit))
	obj, _, err := spotifyRequest(ctx, call, http.MethodGet, spotifyAPI+"/me/player/recently-played?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["items"])
	return tools.Success(string(b), map[string]any{"limit": limit})
}

func (t *RefreshSpotifyTokenTool) Definition() tools.Definition {
	return tools.Definition{Name: "refresh_spotify_token", Description: "Refresh Spotify OAuth access token.", StatusText: "Refreshing Spotify token..."}
}

func (t *RefreshSpotifyTokenTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return refreshOAuthAccessToken(ctx, call, "https://accounts.spotify.com/api/token", true)
}

func spotifyRequest(ctx context.Context, call tools.Call, method, reqURL string, body any) (map[string]any, int, error) {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return nil, 0, err
	}
	token, err := requireToken(ctx, call, cfg)
	if err != nil {
		return nil, 0, fmt.Errorf("Spotify is not connected: %w", err)
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
		return nil, 0, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		if resp.StatusCode == http.StatusNoContent {
			return map[string]any{}, resp.StatusCode, nil
		}
		if resp.StatusCode == http.StatusNotFound {
			return nil, resp.StatusCode, fmt.Errorf("no active Spotify device found")
		}
		if resp.StatusCode == http.StatusForbidden {
			return nil, resp.StatusCode, fmt.Errorf("Spotify premium is required for this action")
		}
		return nil, resp.StatusCode, fmt.Errorf("Spotify API request failed with status %d", resp.StatusCode)
	}
	if resp.StatusCode == http.StatusNoContent {
		return map[string]any{}, resp.StatusCode, nil
	}
	var obj map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&obj); err != nil {
		return nil, resp.StatusCode, err
	}
	return obj, resp.StatusCode, nil
}

var (
	_ tools.Tool = (*NowPlayingTool)(nil)
	_ tools.Tool = (*PlayPauseTool)(nil)
	_ tools.Tool = (*SkipTrackTool)(nil)
	_ tools.Tool = (*SearchSpotifyTool)(nil)
	_ tools.Tool = (*QueueTrackTool)(nil)
	_ tools.Tool = (*RecentTracksTool)(nil)
	_ tools.Tool = (*RefreshSpotifyTokenTool)(nil)
)
