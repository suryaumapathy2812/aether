package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Config holds all environment-based configuration for the agent.
// It is loaded once at startup via Load() and validated via Validate().
type Config struct {
	// Server
	Port      int
	AssetsDir string

	// LLM provider
	LLM LLMConfig

	// S3 / MinIO media storage
	S3 S3Config

	// Web Push (VAPID)
	VAPID VAPIDConfig

	// Channels (Telegram, WhatsApp, etc.)
	Channels ChannelsConfig

	// Agent identity and security
	StateKey     string
	AdminToken   string
	SystemPrompt string
	PromptFile   string
	// Self-update
	UpdateRepo  string
	UpdateToken string

	// Media validation limits
	Media MediaLimitsConfig

	// Proactive engine
	Proactive ProactiveConfig

	// CORS
	CORS CORSConfig
}

// LLMConfig holds LLM provider credentials and model settings.
type LLMConfig struct {
	APIKey            string
	BaseURL           string
	Model             string
	OpenRouterSiteURL string
	OpenRouterAppName string

	// Embedding provider settings (fall back to LLM settings if not set)
	EmbeddingModel   string
	EmbeddingAPIKey  string
	EmbeddingBaseURL string
}

// S3Config holds S3/MinIO connection and behaviour settings.
type S3Config struct {
	Bucket          string
	BucketTemplate  string
	Region          string
	AccessKeyID     string
	SecretAccessKey string
	Endpoint        string
	PublicBaseURL   string
	ForcePathStyle  bool
	PutURLTTL       time.Duration
	GetURLTTL       time.Duration
}

// VAPIDConfig holds Web Push VAPID key settings.
type VAPIDConfig struct {
	PublicKey  string
	PrivateKey string
	Subject    string
}

// ChannelsConfig holds settings for communication channels (Telegram, WhatsApp, etc.)
type ChannelsConfig struct {
	WebhookURL    string // Public URL for webhooks (e.g., from cloudflared)
	WebhookSecret string // Secret token for webhook verification
	AgentID       string // Agent identity used in webhook URL routing (set by orchestrator)
}

// MediaLimitsConfig holds upload/validation size limits.
type MediaLimitsConfig struct {
	MaxImageBytes      int
	MaxAudioBytes      int
	MaxTotalMediaBytes int
	MaxMediaParts      int
}

// ProactiveConfig holds settings for the proactive planning engine.
type ProactiveConfig struct {
	PlanIntervalSeconds int
}

// CORSConfig holds Cross-Origin Resource Sharing settings for the HTTP server.
type CORSConfig struct {
	AllowedOrigins   []string
	AllowedMethods   []string
	AllowedHeaders   []string
	ExposeHeaders    []string
	MaxAge           int
	AllowCredentials bool
}

// Load reads all configuration from environment variables and returns
// a fully populated Config.  It does NOT validate required fields —
// call Validate() for that.
func Load() Config {
	assetsDir := envString("AGENT_ASSETS_DIR", "")
	if assetsDir == "" {
		wd, err := os.Getwd()
		if err != nil {
			assetsDir = "assets"
		} else {
			assetsDir = filepath.Clean(filepath.Join(wd, "assets"))
		}
	}

	// LLM: support OPENAI_* primary, AETHER_LLM_* fallback
	apiKey := firstNonEmpty(envString("OPENAI_API_KEY", ""), envString("AETHER_LLM_API_KEY", ""))
	baseURL := firstNonEmpty(envString("OPENAI_BASE_URL", ""), envString("AETHER_LLM_BASE_URL", ""))
	if baseURL == "" {
		baseURL = "https://openrouter.ai/api/v1"
	}
	baseURL = strings.TrimRight(baseURL, "/")

	model := firstNonEmpty(envString("OPENAI_MODEL", ""), envString("AETHER_LLM_MODEL", ""))
	if model == "" {
		model = "google/gemini-2.5-flash"
	}

	cfg := Config{
		Port:      envInt("PORT", 8000),
		AssetsDir: assetsDir,

		LLM: LLMConfig{
			APIKey:            apiKey,
			BaseURL:           baseURL,
			Model:             model,
			OpenRouterSiteURL: envString("OPENROUTER_SITE_URL", ""),
			OpenRouterAppName: envString("OPENROUTER_APP_NAME", ""),

			// Embedding config - fall back to LLM settings
			EmbeddingModel:   strings.TrimSpace(envString("EMBEDDING_MODEL", "")),
			EmbeddingAPIKey:  strings.TrimSpace(envString("EMBEDDING_API_KEY", "")),
			EmbeddingBaseURL: strings.TrimSpace(envString("EMBEDDING_BASE_URL", "")),
		},

		S3: S3Config{
			Bucket:          envString("S3_BUCKET", ""),
			BucketTemplate:  envString("S3_BUCKET_TEMPLATE", ""),
			Region:          envString("S3_REGION", "us-east-1"),
			AccessKeyID:     envString("S3_ACCESS_KEY_ID", ""),
			SecretAccessKey: envString("S3_SECRET_ACCESS_KEY", ""),
			Endpoint:        envString("S3_ENDPOINT", ""),
			PublicBaseURL:   strings.TrimRight(envString("S3_PUBLIC_BASE_URL", ""), "/"),
			ForcePathStyle:  strings.EqualFold(envString("S3_FORCE_PATH_STYLE", "false"), "true"),
			PutURLTTL:       envDuration("S3_PUT_URL_TTL_SECONDS", 300*time.Second),
			GetURLTTL:       envDuration("S3_GET_URL_TTL_SECONDS", 900*time.Second),
		},

		VAPID: VAPIDConfig{
			PublicKey:  envString("VAPID_PUBLIC_KEY", ""),
			PrivateKey: envString("VAPID_PRIVATE_KEY", ""),
			Subject:    envString("VAPID_SUBJECT", "mailto:admin@aether.local"),
		},

		Channels: ChannelsConfig{
			WebhookURL:    firstNonEmpty(envString("CHANNELS_WEBHOOK_URL", ""), envString("AETHER_PUBLIC_BASE_URL", "")),
			WebhookSecret: envString("CHANNELS_WEBHOOK_SECRET", ""),
			AgentID:       envString("AETHER_AGENT_ID", ""),
		},

		StateKey:     envString("AGENT_STATE_KEY", ""),
		AdminToken:   envString("AGENT_ADMIN_TOKEN", ""),
		SystemPrompt: envString("AGENT_SYSTEM_PROMPT", ""),
		PromptFile:   envString("AGENT_PROMPT_FILE", ""),
		UpdateRepo:   envString("AGENT_UPDATE_REPO", ""),
		UpdateToken:  envString("AGENT_UPDATE_TOKEN", ""),

		Media: MediaLimitsConfig{
			MaxImageBytes:      envInt("AETHER_MAX_IMAGE_BYTES", 5*1024*1024),
			MaxAudioBytes:      envInt("AETHER_MAX_AUDIO_BYTES", 12*1024*1024),
			MaxTotalMediaBytes: envInt("AETHER_MAX_TOTAL_MEDIA_BYTES", 20*1024*1024),
			MaxMediaParts:      envInt("AETHER_MAX_MEDIA_PARTS", 4),
		},

		Proactive: ProactiveConfig{
			PlanIntervalSeconds: envInt("AETHER_PROACTIVE_PLAN_INTERVAL", 10800),
		},

		CORS: CORSConfig{
			AllowedOrigins:   envCSV("CORS_ALLOWED_ORIGINS", []string{"*"}),
			AllowedMethods:   envCSV("CORS_ALLOWED_METHODS", []string{"GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"}),
			AllowedHeaders:   envCSV("CORS_ALLOWED_HEADERS", []string{"Content-Type", "Authorization", "X-Request-ID", "Accept"}),
			ExposeHeaders:    envCSV("CORS_EXPOSE_HEADERS", []string{"X-Request-ID"}),
			MaxAge:           envInt("CORS_MAX_AGE", 86400),
			AllowCredentials: strings.EqualFold(envString("CORS_ALLOW_CREDENTIALS", "false"), "true"),
		},
	}

	// Apply fallback values for embedding config
	if cfg.LLM.EmbeddingModel == "" {
		cfg.LLM.EmbeddingModel = "text-embedding-3-small"
	}
	if cfg.LLM.EmbeddingAPIKey == "" {
		cfg.LLM.EmbeddingAPIKey = cfg.LLM.APIKey
	}
	if cfg.LLM.EmbeddingBaseURL == "" {
		cfg.LLM.EmbeddingBaseURL = cfg.LLM.BaseURL
	}

	return cfg
}

// Validate checks that all required environment variables are present.
// It returns an error listing every missing variable.
func (c Config) Validate() error {
	var missing []string

	// LLM — required for the agent to function
	if c.LLM.APIKey == "" {
		missing = append(missing, "OPENAI_API_KEY (or AETHER_LLM_API_KEY)")
	}

	// S3 — required (media is a core feature)
	if c.S3.Bucket == "" && c.S3.BucketTemplate == "" {
		missing = append(missing, "S3_BUCKET or S3_BUCKET_TEMPLATE (at least one)")
	}
	if c.S3.Bucket != "" || c.S3.BucketTemplate != "" {
		if c.S3.AccessKeyID == "" {
			missing = append(missing, "S3_ACCESS_KEY_ID")
		}
		if c.S3.SecretAccessKey == "" {
			missing = append(missing, "S3_SECRET_ACCESS_KEY")
		}
		if c.S3.Endpoint == "" {
			missing = append(missing, "S3_ENDPOINT")
		}
	}

	if len(missing) == 0 {
		return nil
	}
	return fmt.Errorf("missing required environment variables:\n  - %s", strings.Join(missing, "\n  - "))
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func envString(name, fallback string) string {
	v := strings.TrimSpace(os.Getenv(name))
	v = stripWrappingQuotes(v)
	if v == "" {
		return fallback
	}
	return v
}

func envInt(name string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return fallback
	}
	return v
}

func envDuration(name string, fallback time.Duration) time.Duration {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return fallback
	}
	return time.Duration(n) * time.Second
}

// envCSV reads a comma-separated environment variable and returns a trimmed
// slice of strings.  If the variable is unset or empty, fallback is returned.
func envCSV(name string, fallback []string) []string {
	raw := strings.TrimSpace(os.Getenv(name))
	raw = stripWrappingQuotes(raw)
	if raw == "" {
		return fallback
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if v := strings.TrimSpace(p); v != "" {
			out = append(out, v)
		}
	}
	if len(out) == 0 {
		return fallback
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func stripWrappingQuotes(v string) string {
	v = strings.TrimSpace(v)
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return strings.TrimSpace(v[1 : len(v)-1])
		}
	}
	return v
}
