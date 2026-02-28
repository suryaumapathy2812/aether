package plugins

import "net/http"

import "github.com/suryaumapathy/core-ai/agent/internal/db"

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

type PluginManifest struct {
	Name        string           `yaml:"name" json:"name"`
	DisplayName string           `yaml:"display_name" json:"display_name"`
	Description string           `yaml:"description" json:"description"`
	Version     string           `yaml:"version" json:"version"`
	PluginType  string           `yaml:"plugin_type" json:"plugin_type"`
	Auth        map[string]any   `yaml:"auth" json:"auth"`
	Webhook     map[string]any   `yaml:"webhook" json:"webhook"`
	Tools       []map[string]any `yaml:"tools" json:"tools"`
	Extra       map[string]any   `yaml:",inline" json:"extra,omitempty"`
	Location    string           `yaml:"-" json:"location"`
	Source      SourceType       `yaml:"-" json:"source"`
	SkillPath   string           `yaml:"-" json:"skill_path,omitempty"`
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
