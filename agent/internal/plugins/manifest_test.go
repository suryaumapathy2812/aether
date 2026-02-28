package plugins

import (
	"errors"
	"testing"
)

func TestParseManifest(t *testing.T) {
	raw := []byte("name: gmail\ndisplay_name: Gmail\ndescription: Mail\nversion: 1.0.0\nplugin_type: sensor\nauth:\n  type: oauth2\ntools:\n  - class: SendTool\n")
	m, err := parseManifest(raw)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if m.Name != "gmail" || m.DisplayName != "Gmail" {
		t.Fatalf("unexpected manifest: %#v", m)
	}
	if len(m.Tools) != 1 {
		t.Fatalf("expected 1 tool")
	}
}

func TestParseManifestInvalid(t *testing.T) {
	_, err := parseManifest([]byte("description: no-name"))
	if !errors.Is(err, ErrInvalidPlugin) {
		t.Fatalf("expected invalid plugin error, got %v", err)
	}
}
