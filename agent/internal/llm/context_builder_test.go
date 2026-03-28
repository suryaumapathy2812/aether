package llm

import (
	"strings"
	"testing"
)

func TestNewContextBuilder_AppendsExecutionPolicy(t *testing.T) {
	b := NewContextBuilder(nil, nil, nil, nil, ContextBuilderConfig{})
	prompt := b.SystemPrompt()

	if !strings.Contains(prompt, "Execution policy:") {
		t.Fatalf("expected system prompt to contain execution policy appendix")
	}
}

func TestNewContextBuilder_DoesNotDuplicateArrowContract(t *testing.T) {
	// The Arrow UI contract is defined in PROMPT.md only.
	// Verify it is NOT hardcoded as a Go constant appendix.
	b := NewContextBuilder(nil, nil, nil, nil, ContextBuilderConfig{
		SystemPrompt: "custom prompt without arrow",
	})
	prompt := b.SystemPrompt()

	if strings.Contains(prompt, "Arrow UI contract:") {
		t.Fatalf("Arrow UI contract should not be appended from Go constant; it lives in PROMPT.md")
	}
}

func TestReloadSystemPrompt_PreservesCustomPrompt(t *testing.T) {
	b := NewContextBuilder(nil, nil, nil, nil, ContextBuilderConfig{
		SystemPrompt: "custom prompt",
	})

	reloaded := b.ReloadSystemPrompt()
	if !strings.Contains(reloaded, "custom prompt") {
		t.Fatalf("expected custom prompt to be preserved")
	}
	if !strings.Contains(reloaded, "Execution policy:") {
		t.Fatalf("expected execution policy to be appended on reload")
	}
}
