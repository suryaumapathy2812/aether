package plugins

import (
	"errors"
	"testing"
)

func TestParseManifest(t *testing.T) {
	raw := []byte("name: google-workspace\ndisplay_name: Google Workspace\ndescription: Workspace\nversion: 1.0.0\nplugin_type: sensor\nauth:\n  type: oauth2\n")
	m, err := parseManifest(raw)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if m.Name != "google-workspace" || m.DisplayName != "Google Workspace" {
		t.Fatalf("unexpected manifest: %#v", m)
	}
	if m.Auth.Type != "oauth2" {
		t.Fatalf("expected oauth2 auth type, got %s", m.Auth.Type)
	}
}

func TestParseManifestInvalid(t *testing.T) {
	_, err := parseManifest([]byte("description: no-name"))
	if !errors.Is(err, ErrInvalidPlugin) {
		t.Fatalf("expected invalid plugin error, got %v", err)
	}
}
