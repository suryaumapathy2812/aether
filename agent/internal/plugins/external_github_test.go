package plugins

import "testing"

func TestParseExternalSource(t *testing.T) {
	owner, repo, path, err := parseExternalSource("foo/bar@gmail")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if owner != "foo" || repo != "bar" || path != "gmail" {
		t.Fatalf("unexpected parse: %s %s %s", owner, repo, path)
	}

	owner, repo, path, err = parseExternalSource("foo/bar")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if owner != "foo" || repo != "bar" || path != "" {
		t.Fatalf("unexpected parse: %s %s %s", owner, repo, path)
	}
}

func TestBuildRawURLs(t *testing.T) {
	manifestURL := buildRawManifestURL("https://raw.githubusercontent.com", "o", "r", "")
	if manifestURL != "https://raw.githubusercontent.com/o/r/main/plugin.yaml" {
		t.Fatalf("unexpected url: %s", manifestURL)
	}
	skillURL := buildRawSkillURL("https://raw.githubusercontent.com", "o", "r", "plugins/gmail")
	if skillURL != "https://raw.githubusercontent.com/o/r/main/plugins/gmail/SKILL.md" {
		t.Fatalf("unexpected url: %s", skillURL)
	}
}
