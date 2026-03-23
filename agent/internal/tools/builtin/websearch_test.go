package builtin

import "testing"

func TestParseDDGResults_ParsesLooseWebResultBlocks(t *testing.T) {
	html := `<div class="results"><div><div class="result results_links web-result" data-layout="organic"><a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Farticle">Example <b>Title</b></a><div class="result__snippet">Snippet &amp; details</div></div></div></div>`

	results := parseDDGResults(html)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].title != "Example Title" {
		t.Fatalf("expected title to be stripped, got %q", results[0].title)
	}
	if results[0].url != "https://example.com/article" {
		t.Fatalf("expected decoded URL, got %q", results[0].url)
	}
	if results[0].snippet != "Snippet & details" {
		t.Fatalf("expected decoded snippet, got %q", results[0].snippet)
	}
}

func TestParseDDGResults_FallsBackToResultLinks(t *testing.T) {
	html := `<section><a class="result__a" href="/l/?uddg=https%3A%2F%2Ffallback.example.com">Fallback Result</a><span class="result__snippet">Fallback snippet</span></section>`

	results := parseDDGResults(html)
	if len(results) != 1 {
		t.Fatalf("expected 1 fallback result, got %d", len(results))
	}
	if results[0].title != "Fallback Result" {
		t.Fatalf("expected fallback title, got %q", results[0].title)
	}
	if results[0].url != "https://fallback.example.com" {
		t.Fatalf("expected fallback URL, got %q", results[0].url)
	}
	if results[0].snippet != "Fallback snippet" {
		t.Fatalf("expected fallback snippet, got %q", results[0].snippet)
	}
}
