package config

import (
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	Port                 int
	DatabaseURL          string
	AgentSecret          string
	LocalAgentURL        string
	ProxyTimeout         time.Duration
	DefaultAgentID       string
	AutoAssignFirstAgent bool

	AgentImage         string
	AgentNetwork       string
	AgentIdleTimeout   time.Duration
	AgentPort          int
	AgentHealthTimeout time.Duration

	AgentOpenAIAPIKey  string
	AgentOpenAIBaseURL string
	AgentOpenAIModel   string
	AgentStateKey      string
	VapidPublicKey     string
	VapidPrivateKey    string
	VapidSubject       string
	S3Bucket           string
	S3BucketTemplate   string
	S3Region           string
	S3AccessKeyID      string
	S3SecretAccessKey  string
	S3Endpoint         string
	S3PublicBaseURL    string
	S3ForcePathStyle   string
	S3PutURLTTLSeconds string
	S3GetURLTTLSeconds string
	AgentUpdateRepo    string
	AgentUpdateToken   string
}

func Load() Config {
	port := EnvInt("PORT", 4000)
	proxyTimeout := time.Duration(EnvInt("AGENT_PROXY_TIMEOUT_SECONDS", 120)) * time.Second
	idleTimeout := time.Duration(EnvInt("AGENT_IDLE_TIMEOUT", 1800)) * time.Second
	healthTimeout := time.Duration(EnvInt("AGENT_HEALTH_TIMEOUT", 30)) * time.Second
	databaseURL := strings.TrimSpace(os.Getenv("DATABASE_URL"))
	databaseURL = stripWrappingQuotes(databaseURL)
	if databaseURL == "" {
		log.Fatal("DATABASE_URL is required")
	}

	return Config{
		Port:                 port,
		DatabaseURL:          databaseURL,
		AgentSecret:          stripWrappingQuotes(strings.TrimSpace(os.Getenv("AGENT_SECRET"))),
		LocalAgentURL:        stripWrappingQuotes(strings.TrimSpace(os.Getenv("AETHER_LOCAL_AGENT_URL"))),
		ProxyTimeout:         proxyTimeout,
		DefaultAgentID:       stripWrappingQuotes(strings.TrimSpace(os.Getenv("AETHER_DEFAULT_AGENT_ID"))),
		AutoAssignFirstAgent: strings.EqualFold(strings.TrimSpace(os.Getenv("AETHER_AUTO_ASSIGN_FIRST_AGENT")), "true"),

		AgentImage:         defaultString("AGENT_IMAGE", "suryaumapathy2812/aether-agent:latest"),
		AgentNetwork:       stripWrappingQuotes(strings.TrimSpace(os.Getenv("AGENT_NETWORK"))),
		AgentIdleTimeout:   idleTimeout,
		AgentPort:          EnvInt("AGENT_PORT", 8000),
		AgentHealthTimeout: healthTimeout,

		AgentOpenAIAPIKey:  stripWrappingQuotes(strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))),
		AgentOpenAIBaseURL: stripWrappingQuotes(strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))),
		AgentOpenAIModel:   stripWrappingQuotes(strings.TrimSpace(os.Getenv("OPENAI_MODEL"))),
		AgentStateKey:      stripWrappingQuotes(strings.TrimSpace(os.Getenv("AGENT_STATE_KEY"))),
		VapidPublicKey:     stripWrappingQuotes(strings.TrimSpace(os.Getenv("VAPID_PUBLIC_KEY"))),
		VapidPrivateKey:    stripWrappingQuotes(strings.TrimSpace(os.Getenv("VAPID_PRIVATE_KEY"))),
		VapidSubject:       stripWrappingQuotes(strings.TrimSpace(os.Getenv("VAPID_SUBJECT"))),
		S3Bucket:           stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_BUCKET"))),
		S3BucketTemplate:   stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_BUCKET_TEMPLATE"))),
		S3Region:           stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_REGION"))),
		S3AccessKeyID:      stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_ACCESS_KEY_ID"))),
		S3SecretAccessKey:  stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_SECRET_ACCESS_KEY"))),
		S3Endpoint:         stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_ENDPOINT"))),
		S3PublicBaseURL:    stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_PUBLIC_BASE_URL"))),
		S3ForcePathStyle:   stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_FORCE_PATH_STYLE"))),
		S3PutURLTTLSeconds: stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_PUT_URL_TTL_SECONDS"))),
		S3GetURLTTLSeconds: stripWrappingQuotes(strings.TrimSpace(os.Getenv("S3_GET_URL_TTL_SECONDS"))),
		AgentUpdateRepo:    stripWrappingQuotes(strings.TrimSpace(os.Getenv("AGENT_UPDATE_REPO"))),
		AgentUpdateToken:   stripWrappingQuotes(strings.TrimSpace(os.Getenv("AGENT_UPDATE_TOKEN"))),
	}
}

func EnvInt(name string, fallback int) int {
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

func defaultString(name, fallback string) string {
	v := strings.TrimSpace(os.Getenv(name))
	v = stripWrappingQuotes(v)
	if v == "" {
		return fallback
	}
	return v
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
