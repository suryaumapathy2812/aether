package plugins

import (
	"strings"

	coreplugins "github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// RegisterAvailable reads every discovered plugin manifest and registers its tools.
//
// Tools are registered in one of two ways:
//   - Declarative (new): tool has http.path defined → create an HTTPTool
//   - Legacy (deprecated): tool has class defined → look up Go constructor
//
// Declarative tools are preferred. The legacy path exists only for backward
// compatibility during migration.
func RegisterAvailable(reg *tools.Registry, manager *coreplugins.Manager) error {
	if reg == nil || manager == nil {
		return nil
	}
	legacyCtors := legacyClassConstructors()
	for _, meta := range manager.List() {
		manifest, err := manager.ReadManifest(meta.Name)
		if err != nil {
			continue
		}
		for _, toolDef := range manifest.Tools {
			// Declarative path: tool defines an HTTP endpoint.
			if strings.TrimSpace(toolDef.HTTP.Path) != "" && strings.TrimSpace(toolDef.Name) != "" {
				httpTool := NewHTTPTool(meta.Name, manifest, toolDef)
				if err := reg.Register(httpTool, meta.Name); err != nil {
					return err
				}
				continue
			}

			// Legacy path: tool references a Go class name.
			className := strings.TrimSpace(toolDef.Class)
			if className == "" {
				continue
			}
			ctor, ok := legacyCtors[className]
			if !ok {
				continue
			}
			if err := reg.Register(ctor(), meta.Name); err != nil {
				return err
			}
		}
	}
	return nil
}

// legacyClassConstructors maps legacy class names to Go struct constructors.
// These are only used for plugins that haven't been migrated to declarative manifests yet.
func legacyClassConstructors() map[string]func() tools.Tool {
	return map[string]func() tools.Tool{
		// Weather (uses logic package, not HTTP)
		"CurrentWeatherTool": func() tools.Tool { return &CurrentWeatherTool{} },
		"ForecastTool":       func() tools.Tool { return &ForecastTool{} },
		// Wikipedia (uses logic package)
		"WikipediaSearchTool":     func() tools.Tool { return &WikipediaSearchTool{} },
		"WikipediaGetArticleTool": func() tools.Tool { return &WikipediaGetArticleTool{} },
		// Brave (uses logic package)
		"WebSearchTool":        func() tools.Tool { return &WebSearchTool{} },
		"NewsSearchTool":       func() tools.Tool { return &NewsSearchTool{} },
		"LLMContextSearchTool": func() tools.Tool { return &LLMContextSearchTool{} },
		// Local Search (uses logic package)
		"LocalSearchTool": func() tools.Tool { return &LocalSearchTool{} },
		// RSS (uses logic package)
		"FetchFeedTool":           func() tools.Tool { return &FetchFeedTool{} },
		"GetItemContentTool":      func() tools.Tool { return &GetItemContentTool{} },
		"ListSubscribedFeedsTool": func() tools.Tool { return &ListSubscribedFeedsTool{} },
		"AddFeedTool":             func() tools.Tool { return &AddFeedTool{} },
		"GetHackerNewsTopTool":    func() tools.Tool { return &GetHackerNewsTopTool{} },
		"SearchHackerNewsTool":    func() tools.Tool { return &SearchHackerNewsTool{} },
		// Wolfram (uses logic package)
		"WolframQueryTool":    func() tools.Tool { return &WolframQueryTool{} },
		"CurrencyConvertTool": func() tools.Tool { return &CurrencyConvertTool{} },
	}
}
