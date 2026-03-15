package plugins

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

func parseManifest(raw []byte) (PluginManifest, error) {
	var m PluginManifest
	if err := yaml.Unmarshal(raw, &m); err != nil {
		return PluginManifest{}, fmt.Errorf("%w: %v", ErrInvalidPlugin, err)
	}

	m.Name = strings.TrimSpace(m.Name)
	m.DisplayName = strings.TrimSpace(m.DisplayName)
	m.Description = strings.TrimSpace(m.Description)
	m.Version = strings.TrimSpace(m.Version)
	m.PluginType = strings.TrimSpace(m.PluginType)

	if m.Name == "" {
		return PluginManifest{}, fmt.Errorf("%w: name is required", ErrInvalidPlugin)
	}
	if m.DisplayName == "" {
		m.DisplayName = m.Name
	}
	if m.Version == "" {
		m.Version = "0.1.0"
	}
	if m.PluginType == "" {
		m.PluginType = "sensor"
	}
	if m.Auth.Type == "" {
		m.Auth.Type = "none"
	}
	if m.Webhook == nil {
		m.Webhook = map[string]any{}
	}
	if m.Tools == nil {
		m.Tools = []ManifestTool{}
	}

	// Normalize tool definitions.
	for i := range m.Tools {
		m.Tools[i].Name = strings.TrimSpace(m.Tools[i].Name)
		m.Tools[i].Description = strings.TrimSpace(m.Tools[i].Description)
		m.Tools[i].HTTP.Method = strings.ToUpper(strings.TrimSpace(m.Tools[i].HTTP.Method))
		if m.Tools[i].HTTP.Method == "" {
			m.Tools[i].HTTP.Method = "GET"
		}
	}

	return m, nil
}
