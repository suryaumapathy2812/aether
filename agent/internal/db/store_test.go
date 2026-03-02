package db

import (
	"context"
	"encoding/base64"
	"errors"
	"path/filepath"
	"testing"
)

func TestSkillsCRUD(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	err := store.UpsertSkill(ctx, SkillRecord{Name: "soul", Description: "identity", Location: "assets/skills/builtin/soul/SKILL.md", Source: "builtin"})
	if err != nil {
		t.Fatalf("upsert skill: %v", err)
	}

	got, err := store.GetSkill(ctx, "soul")
	if err != nil {
		t.Fatalf("get skill: %v", err)
	}
	if got.Description != "identity" {
		t.Fatalf("unexpected skill: %#v", got)
	}

	all, err := store.ListSkills(ctx)
	if err != nil {
		t.Fatalf("list skills: %v", err)
	}
	if len(all) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(all))
	}
}

func TestPluginEnabledDefaultsToDisabled(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	err := store.UpsertPlugin(ctx, PluginRecord{
		Name:        "weather",
		DisplayName: "Weather",
		Description: "Forecasts",
		Version:     "0.1.0",
		PluginType:  "sensor",
		Location:    "assets/plugins/builtin/weather/plugin.yaml",
		Source:      "builtin",
	})
	if err != nil {
		t.Fatalf("upsert plugin: %v", err)
	}

	got, err := store.GetPlugin(ctx, "weather")
	if err != nil {
		t.Fatalf("get plugin: %v", err)
	}
	if got.Enabled {
		t.Fatalf("expected default disabled plugin")
	}
}

func TestPluginScopeOnlyOwnRow(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	_ = store.UpsertPlugin(ctx, PluginRecord{Name: "weather", DisplayName: "Weather"})
	_ = store.UpsertPlugin(ctx, PluginRecord{Name: "brave-search", DisplayName: "Brave"})

	weather := store.ScopePlugin("weather")
	if err := weather.SetEnabled(ctx, true); err != nil {
		t.Fatalf("set enabled: %v", err)
	}
	if err := weather.SetConfig(ctx, map[string]string{"api_key": "secret"}); err != nil {
		t.Fatalf("set config: %v", err)
	}

	w, err := store.GetPlugin(ctx, "weather")
	if err != nil {
		t.Fatalf("get weather: %v", err)
	}
	b, err := store.GetPlugin(ctx, "brave-search")
	if err != nil {
		t.Fatalf("get brave: %v", err)
	}

	if !w.Enabled || w.Config["api_key"] != "secret" {
		t.Fatalf("weather scope update missing: %#v", w)
	}
	if b.Enabled || len(b.Config) != 0 {
		t.Fatalf("scope leaked into other plugin: %#v", b)
	}
}

func TestScopeMissingPlugin(t *testing.T) {
	ctx := context.Background()
	store := openTestStore(t)
	defer store.Close()

	scope := store.ScopePlugin("unknown")
	_, err := scope.Get(ctx)
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected not found, got %v", err)
	}
}

func TestEncryptDecryptRoundTrip(t *testing.T) {
	key := []byte("12345678901234567890123456789012")
	stateKey := base64.StdEncoding.EncodeToString(key)

	store := openTestStoreWithKey(t, stateKey)
	defer store.Close()

	enc, err := store.EncryptString("super-secret-token")
	if err != nil {
		t.Fatalf("encrypt: %v", err)
	}
	if enc == "super-secret-token" {
		t.Fatalf("expected encrypted value")
	}

	dec, err := store.DecryptString(enc)
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}
	if dec != "super-secret-token" {
		t.Fatalf("unexpected plaintext: %q", dec)
	}
}

func TestEncryptUnavailableWithoutKey(t *testing.T) {
	t.Setenv("AGENT_STATE_KEY", "")
	store := openTestStore(t)
	defer store.Close()

	_, err := store.EncryptString("x")
	if !errors.Is(err, ErrCryptoUnavailable) {
		t.Fatalf("expected ErrCryptoUnavailable, got %v", err)
	}
}

func TestDecryptInvalidCiphertext(t *testing.T) {
	key := []byte("12345678901234567890123456789012")
	stateKey := base64.StdEncoding.EncodeToString(key)
	store := openTestStoreWithKey(t, stateKey)
	defer store.Close()

	_, err := store.DecryptString("not-encrypted")
	if !errors.Is(err, ErrInvalidCiphertext) {
		t.Fatalf("expected ErrInvalidCiphertext, got %v", err)
	}
}

func openTestStore(t *testing.T) *Store {
	t.Helper()
	return openTestStoreWithKey(t, "")
}

func openTestStoreWithKey(t *testing.T, stateKey string) *Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	store, err := Open(path, stateKey)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	return store
}
