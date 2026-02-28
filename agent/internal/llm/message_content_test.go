package llm

import "testing"

func TestLatestUserTurnSummary_ImageOnlyDoesNotFallback(t *testing.T) {
	messages := []map[string]any{
		{"role": "user", "content": "previous text"},
		{"role": "assistant", "content": "ack"},
		{
			"role": "user",
			"content": []any{
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,aGVsbG8="}},
			},
		},
	}

	got := LatestUserTurnSummary(messages)
	if got != "[image]" {
		t.Fatalf("expected [image], got %q", got)
	}
}

func TestLatestUserTurnSummary_TextWinsWhenPresent(t *testing.T) {
	messages := []map[string]any{{
		"role": "user",
		"content": []any{
			map[string]any{"type": "text", "text": "describe this"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,aGVsbG8="}},
		},
	}}

	got := LatestUserTurnSummary(messages)
	if got != "describe this" {
		t.Fatalf("expected text summary, got %q", got)
	}
}

func TestLatestUserTurnSummary_AudioAndImageMarkers(t *testing.T) {
	messages := []map[string]any{{
		"role": "user",
		"content": []any{
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,aGVsbG8="}},
			map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "aGVsbG8=", "format": "wav"}},
		},
	}}

	got := LatestUserTurnSummary(messages)
	if got != "[image+audio]" {
		t.Fatalf("expected [image+audio], got %q", got)
	}
}
