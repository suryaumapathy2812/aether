package plugins

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type LocalSearchTool struct{}

func (t *LocalSearchTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "local_search",
		Description: "Search for local places near a location using Google Places.",
		StatusText:  "Searching nearby places...",
		Parameters: []tools.Param{
			{Name: "query", Type: "string", Description: "What to search for", Required: true},
			{Name: "location", Type: "string", Description: "Location (city/area)", Required: false},
			{Name: "count", Type: "integer", Description: "Number of results (1-20)", Required: false, Default: 5},
		},
	}
}

func (t *LocalSearchTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	q, _ := call.Args["query"].(string)
	q = strings.TrimSpace(q)
	if q == "" {
		return tools.Fail("Search query is required", nil)
	}
	location, _ := call.Args["location"].(string)
	count, _ := asInt(call.Args["count"])
	if count <= 0 {
		count = 5
	}
	if count > 20 {
		count = 20
	}

	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	googleKey, err := maybeDecryptFromState(ctx, call, strings.TrimSpace(cfg["google_api_key"]))
	if err != nil {
		return tools.Fail("Local search failed to decrypt google_api_key", nil)
	}
	if googleKey == "" {
		googleKey, err = maybeDecryptFromState(ctx, call, strings.TrimSpace(cfg["google_places_api_key"]))
		if err != nil {
			return tools.Fail("Local search failed to decrypt google_places_api_key", nil)
		}
	}
	if googleKey == "" {
		return tools.Fail("Local search requires Google Places API key. Set plugin config key `google_api_key` or `google_places_api_key`.", nil)
	}

	places, err := googleLocalSearch(ctx, googleKey, q, location, count)
	if err != nil {
		return tools.Fail("Local search failed: "+err.Error(), nil)
	}
	if len(places) == 0 {
		return tools.Success("No local results found.", map[string]any{"query": q, "location": location, "count": 0})
	}

	location = strings.TrimSpace(location)
	heading := fmt.Sprintf("Places for '%s':", q)
	if location != "" {
		heading = fmt.Sprintf("Places for '%s' near %s:", q, location)
	}
	lines := []string{heading}
	structured := make([]map[string]any, 0, len(places))
	for i, p := range places {
		rating := ""
		if p.UserRatingCount > 0 {
			rating = fmt.Sprintf("rating %.1f (%d reviews)", p.Rating, p.UserRatingCount)
		}
		meta := strings.TrimSpace(strings.Join([]string{rating, p.Phone}, " | "))
		if meta != "" {
			meta = "\n" + meta
		}
		lines = append(lines, fmt.Sprintf("%d. %s\n%s%s\n%s", i+1, p.Name, p.Address, meta, p.MapsURL))
		structured = append(structured, map[string]any{
			"name":              p.Name,
			"address":           p.Address,
			"rating":            p.Rating,
			"user_rating_count": p.UserRatingCount,
			"maps_url":          p.MapsURL,
			"phone":             p.Phone,
		})
	}
	return tools.Success(strings.Join(lines, "\n\n"), map[string]any{
		"query":    q,
		"location": location,
		"count":    len(places),
		"provider": "google_places",
		"places":   structured,
	})
}

type googlePlace struct {
	Name            string
	Address         string
	Rating          float64
	UserRatingCount int
	MapsURL         string
	Phone           string
}

func googleLocalSearch(ctx context.Context, apiKey, query, location string, count int) ([]googlePlace, error) {
	textQuery := strings.TrimSpace(query)
	if strings.TrimSpace(location) != "" {
		textQuery = fmt.Sprintf("%s near %s", textQuery, strings.TrimSpace(location))
	}
	payload := map[string]any{
		"textQuery":      textQuery,
		"maxResultCount": count,
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://places.googleapis.com/v1/places:searchText", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Goog-Api-Key", apiKey)
	req.Header.Set("X-Goog-FieldMask", "places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.googleMapsUri,places.nationalPhoneNumber")

	httpClient := &http.Client{Timeout: 15 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("google places status %d", resp.StatusCode)
	}

	var data struct {
		Places []struct {
			DisplayName struct {
				Text string `json:"text"`
			} `json:"displayName"`
			FormattedAddress string  `json:"formattedAddress"`
			Rating           float64 `json:"rating"`
			UserRatingCount  int     `json:"userRatingCount"`
			GoogleMapsURI    string  `json:"googleMapsUri"`
			NationalPhone    string  `json:"nationalPhoneNumber"`
		} `json:"places"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	out := make([]googlePlace, 0, len(data.Places))
	for _, p := range data.Places {
		out = append(out, googlePlace{
			Name:            p.DisplayName.Text,
			Address:         p.FormattedAddress,
			Rating:          p.Rating,
			UserRatingCount: p.UserRatingCount,
			MapsURL:         p.GoogleMapsURI,
			Phone:           p.NationalPhone,
		})
	}
	return out, nil
}

var _ tools.Tool = (*LocalSearchTool)(nil)
