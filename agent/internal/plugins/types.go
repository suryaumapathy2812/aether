package plugins

import "net/http"

import "github.com/suryaumapathy2812/core-ai/agent/internal/db"

type SourceType string

const (
	SourceBuiltin  SourceType = "builtin"
	SourceUser     SourceType = "user"
	SourceExternal SourceType = "external"
)

type PluginMeta struct {
	Name        string
	DisplayName string
	Description string
	Version     string
	PluginType  string
	Location    string
	Source      SourceType
	HasSkill    bool
}

// PluginManifest is the parsed representation of a plugin.yaml file.
// It describes a plugin's identity, auth requirements, API configuration, and tools.
type PluginManifest struct {
	Name        string         `yaml:"name" json:"name"`
	DisplayName string         `yaml:"display_name" json:"display_name"`
	Description string         `yaml:"description" json:"description"`
	Version     string         `yaml:"version" json:"version"`
	PluginType  string         `yaml:"plugin_type" json:"plugin_type"`
	Auth        ManifestAuth   `yaml:"auth" json:"auth"`
	API         ManifestAPI    `yaml:"api" json:"api"`
	Webhook     map[string]any `yaml:"webhook" json:"webhook"`
	Tools       []ManifestTool `yaml:"tools" json:"tools"`
	Extra       map[string]any `yaml:",inline" json:"extra,omitempty"`
	Location    string         `yaml:"-" json:"location"`
	Source      SourceType     `yaml:"-" json:"source"`
	SkillPath   string         `yaml:"-" json:"skill_path,omitempty"`
}

// ManifestAuth describes how a plugin authenticates with its external API.
//
// Supported types:
//   - "none"    — no authentication
//   - "api_key" — API key injected as header or query param
//   - "oauth2"  — OAuth2 with token rotation and auto-refresh on 401
//   - "bearer"  — static bearer token from config
type ManifestAuth struct {
	Type            string                `yaml:"type" json:"type"`                     // none | api_key | oauth2 | bearer
	Provider        string                `yaml:"provider" json:"provider,omitempty"`   // e.g. "google", "spotify"
	TokenURL        string                `yaml:"token_url" json:"token_url,omitempty"` // OAuth2 token endpoint
	UseBasicAuth    bool                  `yaml:"use_basic_auth" json:"use_basic_auth,omitempty"`
	AutoRefresh     bool                  `yaml:"auto_refresh" json:"auto_refresh,omitempty"`         // auto-refresh on 401
	RefreshInterval int                   `yaml:"refresh_interval" json:"refresh_interval,omitempty"` // cron rotation seconds
	Scopes          []string              `yaml:"scopes" json:"scopes,omitempty"`
	HeaderName      string                `yaml:"header_name" json:"header_name,omitempty"` // for api_key: header name (default X-Api-Key)
	ConfigKey       string                `yaml:"config_key" json:"config_key,omitempty"`   // config field holding the key/token
	ConfigFields    []ManifestConfigField `yaml:"config_fields" json:"config_fields,omitempty"`
}

// ManifestConfigField describes a user-configurable field for a plugin.
type ManifestConfigField struct {
	Key         string `yaml:"key" json:"key"`
	Label       string `yaml:"label" json:"label"`
	Type        string `yaml:"type" json:"type"` // text | password
	Required    bool   `yaml:"required" json:"required"`
	Description string `yaml:"description" json:"description,omitempty"`
}

// ManifestAPI defines the base URL and default headers for all tools in the plugin.
type ManifestAPI struct {
	BaseURL string            `yaml:"base_url" json:"base_url,omitempty"`
	Headers map[string]string `yaml:"headers" json:"headers,omitempty"`
	Timeout int               `yaml:"timeout" json:"timeout,omitempty"` // seconds, default 20
}

// ManifestTool defines a single tool exposed by a plugin.
type ManifestTool struct {
	Name        string           `yaml:"name" json:"name"`
	Description string           `yaml:"description" json:"description"`
	StatusText  string           `yaml:"status_text" json:"status_text,omitempty"`
	HTTP        ManifestHTTP     `yaml:"http" json:"http"`
	Parameters  []ManifestParam  `yaml:"parameters" json:"parameters,omitempty"`
	Response    ManifestResponse `yaml:"response" json:"response,omitempty"`
	Transform   string           `yaml:"transform" json:"transform,omitempty"` // named transform for complex request building
	Class       string           `yaml:"class" json:"class,omitempty"`         // legacy: Go class name (deprecated, for migration)
}

// ManifestHTTP describes the HTTP call a tool makes.
type ManifestHTTP struct {
	Method  string            `yaml:"method" json:"method"`             // GET | POST | PUT | PATCH | DELETE
	Path    string            `yaml:"path" json:"path"`                 // e.g. /messages/{{message_id}}
	Query   map[string]string `yaml:"query" json:"query,omitempty"`     // static query params
	Body    map[string]any    `yaml:"body" json:"body,omitempty"`       // static body fields
	Headers map[string]string `yaml:"headers" json:"headers,omitempty"` // per-tool headers
}

// ManifestParam defines a parameter for a tool.
type ManifestParam struct {
	Name        string `yaml:"name" json:"name"`
	Type        string `yaml:"type" json:"type"` // string | integer | boolean | array | object
	Description string `yaml:"description" json:"description,omitempty"`
	Required    bool   `yaml:"required" json:"required"`
	Default     any    `yaml:"default" json:"default,omitempty"`
	MapTo       string `yaml:"map_to" json:"map_to,omitempty"` // query.paramName | path.paramName | body.fieldName
}

// ManifestResponse describes how to process the HTTP response.
type ManifestResponse struct {
	Extract    string `yaml:"extract" json:"extract,omitempty"`         // JSON field to extract (e.g. "messages", "items")
	SuccessMsg string `yaml:"success_msg" json:"success_msg,omitempty"` // override success message
}

type InstallResult struct {
	Source     string
	RemoteURL  string
	Installed  PluginMeta
	Downloaded bool
}

type ManagerOptions struct {
	BuiltinDirs []string
	UserDir     string
	ExternalDir string
	HTTPClient  HTTPDoer
	RawBaseURL  string
	StateStore  *db.Store
}

type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}
