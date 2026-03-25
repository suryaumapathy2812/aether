package logic

import "testing"

func TestParseRSSOrAtom_RSS(t *testing.T) {
	raw := []byte(`<?xml version="1.0"?><rss><channel><title>Feed</title><item><title>Hello</title><link>https://x.dev</link><description>World</description><pubDate>Now</pubDate></item></channel></rss>`)
	title, items := parseRSSOrAtom(raw)
	if title != "Feed" {
		t.Fatalf("unexpected title: %q", title)
	}
	if len(items) != 1 || items[0].Title != "Hello" {
		t.Fatalf("unexpected items: %#v", items)
	}
}

func TestParseRSSOrAtom_Atom(t *testing.T) {
	raw := []byte(`<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>A</title><entry><title>T1</title><summary>S1</summary><updated>D1</updated><link href="https://a.dev"/></entry></feed>`)
	title, items := parseRSSOrAtom(raw)
	if title != "A" {
		t.Fatalf("unexpected title: %q", title)
	}
	if len(items) != 1 || items[0].Link != "https://a.dev" {
		t.Fatalf("unexpected items: %#v", items)
	}
}
