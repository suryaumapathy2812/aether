package skills

import "net/http"

type SourceType string

const (
	SourceBuiltin  SourceType = "builtin"
	SourceUser     SourceType = "user"
	SourceExternal SourceType = "external"
)

type SkillMeta struct {
	Name        string
	Description string
	Location    string
	Source      SourceType
}

type Skill struct {
	Meta    SkillMeta
	Content string
}

type InstallResult struct {
	Source     string
	RemoteURL  string
	Installed  SkillMeta
	Downloaded bool
}

type ManagerOptions struct {
	BuiltinDirs []string
	UserDir     string
	ExternalDir string
	HTTPClient  HTTPDoer
	RawBaseURL  string
}

type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}
