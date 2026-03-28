package llm

import (
	"strings"
	"testing"
)

func TestNewContextBuilder_AppendsArrowUIContract(t *testing.T) {
	b := NewContextBuilder(nil, nil, nil, nil, ContextBuilderConfig{})
	prompt := b.SystemPrompt()

	for _, fragment := range []string{
		"Arrow UI contract:",
		"@arrow-js/core",
		"Supported payloads are exactly { type: 'open-url', url: 'https://...' } and { type: 'copy', text: '...' }.",
		"Good interactive example:",
		"Bad examples:",
	} {
		if !strings.Contains(prompt, fragment) {
			t.Fatalf("expected system prompt to contain %q", fragment)
		}
	}
}

func TestReloadSystemPrompt_RetainsArrowUIContract(t *testing.T) {
	b := NewContextBuilder(nil, nil, nil, nil, ContextBuilderConfig{
		SystemPrompt: "custom prompt",
	})

	reloaded := b.ReloadSystemPrompt()
	if !strings.Contains(reloaded, "custom prompt") {
		t.Fatalf("expected custom prompt to be preserved")
	}
	if !strings.Contains(reloaded, "Arrow UI contract:") {
		t.Fatalf("expected Arrow UI contract to be appended on reload")
	}
}
