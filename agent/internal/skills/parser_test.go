package skills

import (
	"errors"
	"testing"
)

func TestParseSkillMarkdown(t *testing.T) {
	raw := "---\nname: soul\ndescription: personality\n---\n\n# Title\nBody"
	parsed, err := parseSkillMarkdown(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if parsed.Name != "soul" {
		t.Fatalf("unexpected name: %q", parsed.Name)
	}
	if parsed.Description != "personality" {
		t.Fatalf("unexpected description: %q", parsed.Description)
	}
	if parsed.Body != "# Title\nBody" {
		t.Fatalf("unexpected body: %q", parsed.Body)
	}
}

func TestParseSkillMarkdownInvalid(t *testing.T) {
	_, err := parseSkillMarkdown("---\nname: only\n---\n")
	if !errors.Is(err, ErrInvalidSkill) {
		t.Fatalf("expected ErrInvalidSkill, got %v", err)
	}
}

func TestSanitizeName(t *testing.T) {
	got := sanitizeName("  My Cool Skill!!! ")
	if got != "my-cool-skill" {
		t.Fatalf("unexpected sanitized name: %q", got)
	}
}
