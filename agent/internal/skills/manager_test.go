package skills

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestManagerDiscoverCreateReadRemove(t *testing.T) {
	root := t.TempDir()
	builtin := filepath.Join(root, "builtin")
	user := filepath.Join(root, "user")
	external := filepath.Join(root, "external")

	mustWriteSkill(t, filepath.Join(builtin, "soul", "SKILL.md"), "soul", "identity", "be real")

	m := NewManager(ManagerOptions{
		BuiltinDirs: []string{builtin},
		UserDir:     user,
		ExternalDir: external,
	})

	if _, err := m.Discover(context.Background()); err != nil {
		t.Fatalf("discover failed: %v", err)
	}

	if _, ok := m.Get("soul"); !ok {
		t.Fatalf("expected soul skill")
	}

	if _, err := m.Create("My Skill", "desc", "hello body"); err != nil {
		t.Fatalf("create failed: %v", err)
	}

	body, err := m.Read("my-skill")
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if body != "hello body" {
		t.Fatalf("unexpected body: %q", body)
	}

	if err := m.Remove("soul"); !errors.Is(err, ErrProtected) {
		t.Fatalf("expected protected error, got %v", err)
	}

	if err := m.Remove("my-skill"); err != nil {
		t.Fatalf("remove user skill failed: %v", err)
	}
	if _, ok := m.Get("my-skill"); ok {
		t.Fatalf("expected removed skill")
	}
}

func TestManagerDuplicateRejected(t *testing.T) {
	root := t.TempDir()
	builtin := filepath.Join(root, "builtin")
	user := filepath.Join(root, "user")

	mustWriteSkill(t, filepath.Join(builtin, "one", "SKILL.md"), "dup", "x", "a")
	mustWriteSkill(t, filepath.Join(user, "two", "SKILL.md"), "dup", "y", "b")

	m := NewManager(ManagerOptions{BuiltinDirs: []string{builtin}, UserDir: user})
	_, err := m.Discover(context.Background())
	if !errors.Is(err, ErrDuplicateName) {
		t.Fatalf("expected duplicate error, got %v", err)
	}
}

func TestManagerSearch(t *testing.T) {
	root := t.TempDir()
	builtin := filepath.Join(root, "builtin")
	mustWriteSkill(t, filepath.Join(builtin, "one", "SKILL.md"), "calendar", "google calendar workflow", "...")
	mustWriteSkill(t, filepath.Join(builtin, "two", "SKILL.md"), "mail", "gmail drafting", "...")

	m := NewManager(ManagerOptions{BuiltinDirs: []string{builtin}})
	if _, err := m.Discover(context.Background()); err != nil {
		t.Fatalf("discover failed: %v", err)
	}

	results := m.Search("calendar google")
	if len(results) != 1 || results[0].Name != "calendar" {
		t.Fatalf("unexpected search results: %#v", results)
	}
}

func TestInstallFromSource(t *testing.T) {
	root := t.TempDir()
	external := filepath.Join(root, "external")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/owner/repo/main/SKILL.md" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte("---\nname: remote-skill\ndescription: from remote\n---\n\ncontent"))
	}))
	defer server.Close()

	m := NewManager(ManagerOptions{ExternalDir: external, RawBaseURL: server.URL})
	if _, err := m.Discover(context.Background()); err != nil {
		t.Fatalf("discover failed: %v", err)
	}

	result, err := m.InstallFromSource(context.Background(), "owner/repo")
	if err != nil {
		t.Fatalf("install failed: %v", err)
	}
	if result.Installed.Name != "remote-skill" {
		t.Fatalf("unexpected installed skill: %#v", result)
	}

	if _, ok := m.Get("remote-skill"); !ok {
		t.Fatalf("installed skill not indexed")
	}

	if _, err := os.Stat(filepath.Join(external, "remote-skill", "SKILL.md")); err != nil {
		t.Fatalf("expected installed file: %v", err)
	}
}

func TestAttachDirectory(t *testing.T) {
	root := t.TempDir()
	extra := filepath.Join(root, "extra")
	mustWriteSkill(t, filepath.Join(extra, "x", "SKILL.md"), "external-skill", "desc", "body")

	m := NewManager(ManagerOptions{})
	if err := m.AttachDirectory(extra, SourceExternal); err != nil {
		t.Fatalf("attach failed: %v", err)
	}

	if _, err := m.Discover(context.Background()); err != nil {
		t.Fatalf("discover failed: %v", err)
	}

	if _, ok := m.Get("external-skill"); !ok {
		t.Fatalf("attached skill missing")
	}
}

func mustWriteSkill(t *testing.T, path, name, desc, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	content := buildSkillMarkdown(name, desc, body)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write skill failed: %v", err)
	}
}
