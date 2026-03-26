package tools

import (
	"embed"
	"encoding/json"
	"strings"
)

//go:embed sandbox-templates
var templateFS embed.FS

// SandboxTemplate is an Arrow sandbox template that renders tool output.
type SandboxTemplate struct {
	Source map[string]string
}

// sandboxRegistry maps tool names → their Arrow sandbox templates.
var sandboxRegistry = map[string]SandboxTemplate{}

// RegisterSandbox registers a sandbox template for a tool name.
func RegisterSandbox(toolName string, tmpl SandboxTemplate) {
	sandboxRegistry[toolName] = tmpl
}

// GetSandbox returns the sandbox template for a tool name, or nil.
func GetSandbox(toolName string) *SandboxTemplate {
	tmpl, ok := sandboxRegistry[toolName]
	if !ok {
		return nil
	}
	return &tmpl
}

// GenerateSandboxSource fills a template's {{DATA}} placeholder with the
// tool output JSON. Returns nil if the tool has no template.
func GenerateSandboxSource(toolName string, output string) map[string]string {
	tmpl := GetSandbox(toolName)
	if tmpl == nil {
		return nil
	}

	escaped := escapeForJSTemplate(output)

	source := make(map[string]string, len(tmpl.Source))
	for filename, content := range tmpl.Source {
		source[filename] = strings.ReplaceAll(content, "{{DATA}}", escaped)
	}
	return source
}

// escapeForJSTemplate escapes a string for embedding inside a JS template literal.
func escapeForJSTemplate(s string) string {
	r := strings.NewReplacer(
		"\\", "\\\\",
		"`", "\\`",
		"${", "\\${",
	)
	return r.Replace(s)
}

// WrapWithSandbox checks if a sandbox template exists for the given tool
// and wraps the result metadata accordingly.
func WrapWithSandbox(toolName string, result Result) Result {
	source := GenerateSandboxSource(toolName, result.Output)
	if source == nil {
		return result
	}
	return result.WithSandbox(source)
}

// --- Template loading via go:embed ---------------------------------------

func init() {
	templates := map[string]string{
		// Gmail
		"inbox_count":  "gmail/inbox-count.ts",
		"list_unread":  "gmail/list.ts",
		"search_email": "gmail/list.ts",
		"read_gmail":   "gmail/read.ts",
		"send_email":   "gmail/sent.ts",
		"send_reply":   "gmail/sent.ts",
		"create_draft": "gmail/sent.ts",
		// Calendar
		"upcoming_events": "calendar/list.ts",
		"search_events":   "calendar/list.ts",
		"get_event":       "calendar/event.ts",
		"create_event":    "calendar/created.ts",
		// Drive
		"search_drive":     "drive/list.ts",
		"list_drive_files": "drive/list.ts",
		// Contacts
		"search_contacts": "contacts/list.ts",
		"get_contact":     "contacts/detail.ts",
		// Web
		"web_search":   "web/search.ts",
		"local_search": "web/local.ts",
	}

	for toolName, filePath := range templates {
		content, err := templateFS.ReadFile("sandbox-templates/" + filePath)
		if err != nil {
			continue
		}
		RegisterSandbox(toolName, SandboxTemplate{
			Source: map[string]string{"main.ts": string(content)},
		})
	}
}

// jsonIndent is a debug helper.
func jsonIndent(v any) string {
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}
