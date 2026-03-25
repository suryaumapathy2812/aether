package plugins

import "testing"

func TestParseExternalSource(t *testing.T) {
	owner, repo, path, err := parseExternalSource("foo/bar@google-workspace")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if owner != "foo" || repo != "bar" || path != "google-workspace" {
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
	skillURL := buildRawSkillURL("https://raw.githubusercontent.com", "o", "r", "plugins/google-workspace")
	if skillURL != "https://raw.githubusercontent.com/o/r/main/plugins/google-workspace/SKILL.md" {
		t.Fatalf("unexpected url: %s", skillURL)
	}
}
