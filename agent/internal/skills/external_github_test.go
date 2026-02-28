package skills

import "testing"

func TestParseExternalSource(t *testing.T) {
	owner, repo, skill, err := parseExternalSource("foo/bar@baz")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if owner != "foo" || repo != "bar" || skill != "baz" {
		t.Fatalf("unexpected parse: %s %s %s", owner, repo, skill)
	}

	owner, repo, skill, err = parseExternalSource("foo/bar")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if owner != "foo" || repo != "bar" || skill != "" {
		t.Fatalf("unexpected parse: %s %s %s", owner, repo, skill)
	}
}

func TestBuildRawURL(t *testing.T) {
	if got := buildRawURL("https://raw.githubusercontent.com", "o", "r", ""); got != "https://raw.githubusercontent.com/o/r/main/SKILL.md" {
		t.Fatalf("unexpected url: %s", got)
	}
	if got := buildRawURL("https://raw.githubusercontent.com/", "o", "r", "skill"); got != "https://raw.githubusercontent.com/o/r/main/skill/SKILL.md" {
		t.Fatalf("unexpected url: %s", got)
	}
}
