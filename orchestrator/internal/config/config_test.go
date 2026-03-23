package config

import (
	"os"
	"testing"
	"time"
)

func TestLoadReadsDirectRoutingOverrides(t *testing.T) {
	t.Setenv("DATABASE_URL", "postgres://example")
	t.Setenv("AGENT_SECRET", "secret")
	t.Setenv("AGENT_DIRECT_TOKEN_SECRET", "direct")
	t.Setenv("OPENAI_API_KEY", "key")
	t.Setenv("S3_BUCKET", "bucket")
	t.Setenv("S3_ACCESS_KEY_ID", "access")
	t.Setenv("S3_SECRET_ACCESS_KEY", "secret")
	t.Setenv("S3_ENDPOINT", "http://minio:9000")
	t.Setenv("DIRECT_AGENT_DOMAIN", "agents.example.com")
	t.Setenv("CADDY_ADMIN_URL", "http://caddy:2019")
	t.Setenv("AGENT_RECONCILE_INTERVAL", "45")

	cfg := Load()

	if cfg.DirectAgentDomain != "agents.example.com" {
		t.Fatalf("expected direct agent domain override, got %q", cfg.DirectAgentDomain)
	}
	if cfg.CaddyAdminURL != "http://caddy:2019" {
		t.Fatalf("expected caddy admin url override, got %q", cfg.CaddyAdminURL)
	}
	if cfg.AgentReconcileInterval != 45*time.Second {
		t.Fatalf("expected 45s reconcile interval, got %s", cfg.AgentReconcileInterval)
	}
}

func TestLoadFallsBackToDefaultReconcileInterval(t *testing.T) {
	for _, key := range []string{
		"DIRECT_AGENT_DOMAIN",
		"CADDY_ADMIN_URL",
		"AGENT_RECONCILE_INTERVAL",
	} {
		if err := os.Unsetenv(key); err != nil {
			t.Fatalf("unset %s: %v", key, err)
		}
	}
	t.Setenv("DATABASE_URL", "postgres://example")
	t.Setenv("AGENT_SECRET", "secret")
	t.Setenv("AGENT_DIRECT_TOKEN_SECRET", "direct")
	t.Setenv("OPENAI_API_KEY", "key")
	t.Setenv("S3_BUCKET", "bucket")
	t.Setenv("S3_ACCESS_KEY_ID", "access")
	t.Setenv("S3_SECRET_ACCESS_KEY", "secret")
	t.Setenv("S3_ENDPOINT", "http://minio:9000")

	cfg := Load()

	if cfg.DirectAgentDomain != DefaultDirectAgentDomain {
		t.Fatalf("expected default direct agent domain, got %q", cfg.DirectAgentDomain)
	}
	if cfg.CaddyAdminURL != DefaultCaddyAdminURL {
		t.Fatalf("expected default caddy admin url, got %q", cfg.CaddyAdminURL)
	}
	if cfg.AgentReconcileInterval != DefaultReconcileInterval {
		t.Fatalf("expected default reconcile interval, got %s", cfg.AgentReconcileInterval)
	}
}
