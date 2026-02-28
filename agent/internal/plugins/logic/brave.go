package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
)

const braveBase = "https://api.search.brave.com/res/v1"

type BraveClient struct {
	HTTP   HTTPClient
	APIKey string
}

func (b BraveClient) WebSearch(ctx context.Context, query string, count int, country string) ([]WebResult, error) {
	if count < 1 {
		count = 1
	}
	if count > 20 {
		count = 20
	}
	if country == "" {
		country = "IN"
	}
	v := url.Values{}
	v.Set("q", query)
	v.Set("count", fmt.Sprintf("%d", count))
	v.Set("country", country)
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, braveBase+"/web/search?"+v.Encode(), nil)
	b.setHeaders(req)
	resp, err := defaultHTTPClient(b.HTTP).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("brave web status: %d", resp.StatusCode)
	}
	var payload struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	out := make([]WebResult, 0, len(payload.Web.Results))
	for _, r := range payload.Web.Results {
		out = append(out, WebResult{Title: r.Title, URL: r.URL, Description: r.Description})
	}
	return out, nil
}

func (b BraveClient) NewsSearch(ctx context.Context, query string, count int, country string) ([]NewsResult, error) {
	if count < 1 {
		count = 1
	}
	if count > 20 {
		count = 20
	}
	if country == "" {
		country = "IN"
	}
	v := url.Values{}
	v.Set("q", query)
	v.Set("count", fmt.Sprintf("%d", count))
	v.Set("country", country)
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, braveBase+"/news/search?"+v.Encode(), nil)
	b.setHeaders(req)
	resp, err := defaultHTTPClient(b.HTTP).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("brave news status: %d", resp.StatusCode)
	}
	var payload struct {
		Results []struct {
			Title       string `json:"title"`
			URL         string `json:"url"`
			Description string `json:"description"`
			MetaURL     struct {
				Hostname string `json:"hostname"`
			} `json:"meta_url"`
			Age string `json:"age"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	out := make([]NewsResult, 0, len(payload.Results))
	for _, r := range payload.Results {
		out = append(out, NewsResult{Title: r.Title, URL: r.URL, Description: r.Description, Source: r.MetaURL.Hostname, Age: r.Age})
	}
	return out, nil
}

func (b BraveClient) LLMContextSearch(ctx context.Context, query string) ([]ContextSnippet, error) {
	v := url.Values{}
	v.Set("q", query)
	v.Set("count", "5")
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, braveBase+"/web/search?"+v.Encode(), nil)
	b.setHeaders(req)
	req.Header.Set("X-Respond-With", "llm_context")
	resp, err := defaultHTTPClient(b.HTTP).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("brave context status: %d", resp.StatusCode)
	}
	var payload struct {
		LLMContext struct {
			Snippets []struct {
				Title string `json:"title"`
				URL   string `json:"url"`
				Text  string `json:"text"`
			} `json:"snippets"`
		} `json:"llm_context"`
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	out := []ContextSnippet{}
	for _, s := range payload.LLMContext.Snippets {
		if s.Text != "" {
			out = append(out, ContextSnippet{Title: s.Title, URL: s.URL, Text: s.Text})
		}
	}
	if len(out) > 0 {
		return out, nil
	}
	for _, r := range payload.Web.Results {
		if r.Description != "" {
			out = append(out, ContextSnippet{Title: r.Title, URL: r.URL, Text: r.Description})
		}
	}
	return out, nil
}

func (b BraveClient) setHeaders(req *http.Request) {
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Accept-Encoding", "gzip")
	req.Header.Set("X-Subscription-Token", b.APIKey)
}
