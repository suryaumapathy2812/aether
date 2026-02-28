package logic

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
)

const (
	hnFirebase = "https://hacker-news.firebaseio.com/v0"
	hnAlgolia  = "https://hn.algolia.com/api/v1"
)

type RSSClient struct{ HTTP HTTPClient }

type FeedItem struct {
	Title   string
	Link    string
	Summary string
	PubDate string
}

type HNStory struct {
	ID       int
	Title    string
	URL      string
	Score    int
	Comments int
}

func (r RSSClient) FetchFeed(ctx context.Context, feedURL string, maxItems int) ([]FeedItem, string, error) {
	if maxItems < 1 {
		maxItems = 1
	}
	if maxItems > 20 {
		maxItems = 20
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, feedURL, nil)
	req.Header.Set("User-Agent", "Aether-Go-Agent/1.0")
	resp, err := defaultHTTPClient(r.HTTP).Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("feed status: %d", resp.StatusCode)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", err
	}

	title, items := parseRSSOrAtom(b)
	if len(items) > maxItems {
		items = items[:maxItems]
	}
	return items, title, nil
}

func (r RSSClient) GetItemContent(ctx context.Context, pageURL string) (string, string, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, pageURL, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Aether-Go-Agent/1.0)")
	resp, err := defaultHTTPClient(r.HTTP).Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("article status: %d", resp.StatusCode)
	}
	htmlBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", "", err
	}
	html := string(htmlBytes)
	title := extractTag(html, "title")
	content := bestEffortArticleText(html)
	if content == "" {
		return title, "", fmt.Errorf("could not extract readable content")
	}
	return title, content, nil
}

func (r RSSClient) GetHNTop(ctx context.Context, storyType string, count int) ([]HNStory, error) {
	if count < 1 {
		count = 1
	}
	if count > 30 {
		count = 30
	}
	endpoint := map[string]string{"top": "topstories", "new": "newstories", "best": "beststories", "ask": "askstories", "show": "showstories"}[storyType]
	if endpoint == "" {
		endpoint = "topstories"
	}
	idReq, _ := http.NewRequestWithContext(ctx, http.MethodGet, hnFirebase+"/"+endpoint+".json", nil)
	idResp, err := defaultHTTPClient(r.HTTP).Do(idReq)
	if err != nil {
		return nil, err
	}
	defer idResp.Body.Close()
	if idResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("hn ids status: %d", idResp.StatusCode)
	}
	var ids []int
	if err := json.NewDecoder(idResp.Body).Decode(&ids); err != nil {
		return nil, err
	}
	if len(ids) > count {
		ids = ids[:count]
	}

	out := make([]HNStory, 0, len(ids))
	for _, id := range ids {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/item/%d.json", hnFirebase, id), nil)
		resp, err := defaultHTTPClient(r.HTTP).Do(req)
		if err != nil {
			continue
		}
		var s struct {
			ID          int    `json:"id"`
			Type        string `json:"type"`
			Title       string `json:"title"`
			URL         string `json:"url"`
			Score       int    `json:"score"`
			Descendants int    `json:"descendants"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&s)
		resp.Body.Close()
		if s.Type == "story" || s.Type == "ask" || s.Type == "show" {
			out = append(out, HNStory{ID: s.ID, Title: s.Title, URL: s.URL, Score: s.Score, Comments: s.Descendants})
		}
	}
	return out, nil
}

func (r RSSClient) SearchHN(ctx context.Context, query string, count int, searchType string) ([]HNStory, error) {
	if count < 1 {
		count = 1
	}
	if count > 20 {
		count = 20
	}
	tags := "story"
	if strings.ToLower(strings.TrimSpace(searchType)) == "all" {
		tags = "(story,comment)"
	}
	v := url.Values{}
	v.Set("query", query)
	v.Set("tags", tags)
	v.Set("hitsPerPage", fmt.Sprintf("%d", count))
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, hnAlgolia+"/search_by_date?"+v.Encode(), nil)
	resp, err := defaultHTTPClient(r.HTTP).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("hn search status: %d", resp.StatusCode)
	}
	var payload struct {
		Hits []struct {
			ObjectID    string `json:"objectID"`
			Title       string `json:"title"`
			StoryTitle  string `json:"story_title"`
			URL         string `json:"url"`
			Points      int    `json:"points"`
			NumComments int    `json:"num_comments"`
		} `json:"hits"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	out := make([]HNStory, 0, len(payload.Hits))
	for _, h := range payload.Hits {
		t := h.Title
		if t == "" {
			t = h.StoryTitle
		}
		out = append(out, HNStory{Title: t, URL: h.URL, Score: h.Points, Comments: h.NumComments})
	}
	return out, nil
}

func parseRSSOrAtom(raw []byte) (string, []FeedItem) {
	type rss struct {
		Channel struct {
			Title string `xml:"title"`
			Items []struct {
				Title       string `xml:"title"`
				Link        string `xml:"link"`
				Description string `xml:"description"`
				PubDate     string `xml:"pubDate"`
			} `xml:"item"`
		} `xml:"channel"`
	}
	type atom struct {
		Title string `xml:"title"`
		Entry []struct {
			Title   string `xml:"title"`
			Summary string `xml:"summary"`
			Updated string `xml:"updated"`
			Link    struct {
				Href string `xml:"href,attr"`
			} `xml:"link"`
		} `xml:"entry"`
	}

	var r rss
	if err := xml.Unmarshal(raw, &r); err == nil && len(r.Channel.Items) > 0 {
		items := make([]FeedItem, 0, len(r.Channel.Items))
		for _, it := range r.Channel.Items {
			items = append(items, FeedItem{Title: stripTags(it.Title), Link: strings.TrimSpace(it.Link), Summary: stripTags(it.Description), PubDate: strings.TrimSpace(it.PubDate)})
		}
		return stripTags(r.Channel.Title), items
	}

	var a atom
	if err := xml.Unmarshal(raw, &a); err == nil && len(a.Entry) > 0 {
		items := make([]FeedItem, 0, len(a.Entry))
		for _, e := range a.Entry {
			items = append(items, FeedItem{Title: stripTags(e.Title), Link: strings.TrimSpace(e.Link.Href), Summary: stripTags(e.Summary), PubDate: strings.TrimSpace(e.Updated)})
		}
		return stripTags(a.Title), items
	}
	return "", nil
}

var tagRE = regexp.MustCompile(`<[^>]+>`)

func stripTags(s string) string {
	if s == "" {
		return ""
	}
	s = tagRE.ReplaceAllString(s, "")
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&#39;", "'")
	return strings.TrimSpace(s)
}

func extractTag(html, tag string) string {
	re := regexp.MustCompile(`(?is)<` + tag + `[^>]*>(.*?)</` + tag + `>`)
	m := re.FindStringSubmatch(html)
	if len(m) < 2 {
		return ""
	}
	return stripTags(m[1])
}

func bestEffortArticleText(html string) string {
	patterns := []string{
		`(?is)<article[^>]*>(.*?)</article>`,
		`(?is)<main[^>]*>(.*?)</main>`,
		`(?is)<div[^>]+class=["'][^"']*(?:content|article|post|entry)[^"']*["'][^>]*>(.*?)</div>`,
	}
	for _, p := range patterns {
		re := regexp.MustCompile(p)
		m := re.FindStringSubmatch(html)
		if len(m) >= 2 {
			text := stripTags(m[1])
			if len(text) > 200 {
				return normalizeWhitespace(text)
			}
		}
	}
	return ""
}

func normalizeWhitespace(s string) string {
	parts := strings.Fields(s)
	return strings.Join(parts, " ")
}

func sortByScore(stories []HNStory) {
	sort.Slice(stories, func(i, j int) bool { return stories[i].Score > stories[j].Score })
}
