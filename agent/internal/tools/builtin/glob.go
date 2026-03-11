package builtin

import (
	"context"
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// GlobTool finds files matching a glob pattern within the working directory.
type GlobTool struct{}

func (t *GlobTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "glob",
		Description: "Find files matching a glob pattern within the working directory.",
		StatusText:  "Searching files...",
		Parameters: []tools.Param{
			{Name: "pattern", Type: "string", Description: "Glob pattern like **/*.go, src/**/*.ts", Required: true},
			{Name: "path", Type: "string", Description: "Subdirectory to search in", Required: false, Default: "."},
		},
	}
}

func (t *GlobTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	pattern, _ := call.Args["pattern"].(string)
	searchPath, _ := call.Args["path"].(string)
	if searchPath == "" {
		searchPath = "."
	}

	resolved, err := resolveSafePath(call.Ctx.WorkingDir, searchPath)
	if err != nil {
		return tools.Fail("Invalid path: "+err.Error(), nil)
	}

	const maxResults = 1000
	var matches []string

	// Check if pattern contains ** (doublestar) which filepath.Glob doesn't support.
	if strings.Contains(pattern, "**") {
		matches, err = globDoublestar(resolved, pattern, maxResults+1)
	} else {
		matches, err = globSimple(resolved, pattern, maxResults+1)
	}
	if err != nil {
		return tools.Fail("Glob error: "+err.Error(), nil)
	}

	sort.Strings(matches)

	truncated := false
	if len(matches) > maxResults {
		matches = matches[:maxResults]
		truncated = true
	}

	output := strings.Join(matches, "\n")
	if truncated {
		output += "\n...truncated"
	}

	return tools.Success(output, map[string]any{"count": len(matches)})
}

// globSimple handles patterns without ** using filepath.Match on each file.
func globSimple(root, pattern string, limit int) ([]string, error) {
	var matches []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip inaccessible entries
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		// Match against the relative path's base name or full relative path
		// depending on whether the pattern contains a separator.
		matchTarget := rel
		if !strings.Contains(pattern, string(filepath.Separator)) && !strings.Contains(pattern, "/") {
			matchTarget = filepath.Base(rel)
		}
		matched, err := filepath.Match(pattern, matchTarget)
		if err != nil {
			return fmt.Errorf("invalid pattern: %w", err)
		}
		if matched {
			matches = append(matches, rel)
			if len(matches) >= limit {
				return filepath.SkipAll
			}
		}
		return nil
	})
	return matches, err
}

// globDoublestar handles patterns with ** by walking the directory tree
// and matching each path segment-by-segment.
func globDoublestar(root, pattern string, limit int) ([]string, error) {
	// Normalize pattern separators to OS separator.
	pattern = filepath.FromSlash(pattern)
	patternParts := strings.Split(pattern, string(filepath.Separator))

	var matches []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip inaccessible entries
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		relParts := strings.Split(rel, string(filepath.Separator))
		if matchDoublestar(patternParts, relParts) {
			matches = append(matches, rel)
			if len(matches) >= limit {
				return filepath.SkipAll
			}
		}
		return nil
	})
	return matches, err
}

// matchDoublestar matches a path (split into parts) against a pattern (split into parts)
// where ** matches zero or more directory segments.
func matchDoublestar(pattern, path []string) bool {
	return matchDS(pattern, path, 0, 0)
}

func matchDS(pattern, path []string, pi, si int) bool {
	for pi < len(pattern) && si < len(path) {
		if pattern[pi] == "**" {
			// ** can match zero or more segments.
			// Try matching the rest of the pattern starting from every
			// remaining position in the path (including current).
			for k := si; k <= len(path); k++ {
				if matchDS(pattern, path, pi+1, k) {
					return true
				}
			}
			return false
		}
		matched, err := filepath.Match(pattern[pi], path[si])
		if err != nil || !matched {
			return false
		}
		pi++
		si++
	}
	// Consume trailing ** patterns (they can match zero segments).
	for pi < len(pattern) && pattern[pi] == "**" {
		pi++
	}
	return pi == len(pattern) && si == len(path)
}

var _ tools.Tool = (*GlobTool)(nil)
