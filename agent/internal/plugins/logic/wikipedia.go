package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
)

const wikipediaREST = "https://en.wikipedia.org/api/rest_v1"

type WikipediaClient struct{ HTTP HTTPClient }

type WikipediaSummary struct {
	Title       string
	Description string
	Extract     string
	PageURL     string
}

func (w WikipediaClient) SearchSummary(ctx context.Context, query string) (WikipediaSummary, error) {
	return w.summaryByTitle(ctx, query, true)
}

func (w WikipediaClient) GetArticle(ctx context.Context, title string) (WikipediaSummary, error) {
	return w.summaryByTitle(ctx, title, false)
}

func (w WikipediaClient) summaryByTitle(ctx context.Context, titleOrQuery string, fallbackSearch bool) (WikipediaSummary, error) {
	encoded := url.PathEscape(titleOrQuery)
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, wikipediaREST+"/page/summary/"+encoded, nil)
	req.Header.Set("User-Agent", "Aether-Go-Agent/1.0")
	resp, err := defaultHTTPClient(w.HTTP).Do(req)
	if err != nil {
		return WikipediaSummary{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound && fallbackSearch {
		title, err := w.searchTitle(ctx, titleOrQuery)
		if err != nil {
			return WikipediaSummary{}, err
		}
		return w.summaryByTitle(ctx, title, false)
	}
	if resp.StatusCode != http.StatusOK {
		return WikipediaSummary{}, fmt.Errorf("wikipedia status: %d", resp.StatusCode)
	}

	var payload struct {
		Title       string `json:"title"`
		Description string `json:"description"`
		Extract     string `json:"extract"`
		ContentURLs struct {
			Desktop struct {
				Page string `json:"page"`
			} `json:"desktop"`
		} `json:"content_urls"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return WikipediaSummary{}, err
	}
	return WikipediaSummary{
		Title:       payload.Title,
		Description: payload.Description,
		Extract:     payload.Extract,
		PageURL:     payload.ContentURLs.Desktop.Page,
	}, nil
}

func (w WikipediaClient) searchTitle(ctx context.Context, query string) (string, error) {
	v := url.Values{}
	v.Set("action", "query")
	v.Set("list", "search")
	v.Set("srsearch", query)
	v.Set("format", "json")
	v.Set("srlimit", "1")
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "https://en.wikipedia.org/w/api.php?"+v.Encode(), nil)
	req.Header.Set("User-Agent", "Aether-Go-Agent/1.0")
	resp, err := defaultHTTPClient(w.HTTP).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("wikipedia search status: %d", resp.StatusCode)
	}
	var payload struct {
		Query struct {
			Search []struct {
				Title string `json:"title"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", err
	}
	if len(payload.Query.Search) == 0 {
		return "", fmt.Errorf("no wikipedia results for: %s", query)
	}
	return payload.Query.Search[0].Title, nil
}
