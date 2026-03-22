package skills

import "net/http"

type SourceType string

const (
	SourceBuiltin  SourceType = "builtin"
	SourceUser     SourceType = "user"
	SourceExternal SourceType = "external"
)

type SkillMeta struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	AlwaysLoad  bool       `json:"always_load"`
	Location    string     `json:"location"`
	Source      SourceType `json:"source"`
}

type Skill struct {
	Meta    SkillMeta
	Content string
}

type MarketplaceSkill struct {
	ID       string `json:"id"`
	SkillID  string `json:"skill_id"`
	Name     string `json:"name"`
	Installs int    `json:"installs"`
	Source   string `json:"source"`
}

type MarketplaceSearchResult struct {
	Query      string             `json:"query"`
	SearchType string             `json:"search_type"`
	Skills     []MarketplaceSkill `json:"skills"`
	Count      int                `json:"count"`
	DurationMS int                `json:"duration_ms"`
}

type InstallResult struct {
	Source     string    `json:"source"`
	RemoteURL  string    `json:"remote_url"`
	Installed  SkillMeta `json:"installed"`
	Downloaded bool      `json:"downloaded"`
}

type ManagerOptions struct {
	BuiltinDirs []string
	UserDir     string
	ExternalDir string
	HTTPClient  HTTPDoer
	RawBaseURL  string
	SkillsAPI   string
}

type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}
