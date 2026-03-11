package builtin

import (
	"bufio"
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// GrepTool searches file contents using a regular expression pattern.
type GrepTool struct{}

func (t *GrepTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "grep",
		Description: "Search file contents using a regular expression pattern.",
		StatusText:  "Searching content...",
		Parameters: []tools.Param{
			{Name: "pattern", Type: "string", Description: "Regex pattern to search for", Required: true},
			{Name: "path", Type: "string", Description: "Directory to search in", Required: false, Default: "."},
			{Name: "include", Type: "string", Description: "File glob filter like *.go, *.{ts,tsx}", Required: false},
		},
	}
}

// skipDirs contains directory names that should be skipped during search.
var skipDirs = map[string]bool{
	".git":         true,
	"node_modules": true,
	"vendor":       true,
}

func (t *GrepTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	pattern, _ := call.Args["pattern"].(string)
	searchPath, _ := call.Args["path"].(string)
	include, _ := call.Args["include"].(string)
	if searchPath == "" {
		searchPath = "."
	}

	resolved, err := resolveSafePath(call.Ctx.WorkingDir, searchPath)
	if err != nil {
		return tools.Fail("Invalid path: "+err.Error(), nil)
	}

	re, err := regexp.Compile(pattern)
	if err != nil {
		return tools.Fail("Invalid regex pattern: "+err.Error(), nil)
	}

	// Expand brace patterns like *.{ts,tsx} into multiple glob patterns.
	includePatterns := expandBraces(include)

	const maxMatches = 500
	var results []string
	fileSet := map[string]bool{}

	err = filepath.WalkDir(resolved, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip inaccessible entries
		}
		if d.IsDir() {
			name := d.Name()
			// Skip hidden directories and common large directories.
			if strings.HasPrefix(name, ".") && name != "." {
				return filepath.SkipDir
			}
			if skipDirs[name] {
				return filepath.SkipDir
			}
			return nil
		}

		// Apply include filter if specified.
		if len(includePatterns) > 0 {
			base := d.Name()
			matched := false
			for _, p := range includePatterns {
				if m, _ := filepath.Match(p, base); m {
					matched = true
					break
				}
			}
			if !matched {
				return nil
			}
		}

		// Skip binary files by checking first 512 bytes for null bytes.
		if isBinaryFile(path) {
			return nil
		}

		rel, err := filepath.Rel(call.Ctx.WorkingDir, path)
		if err != nil {
			rel = path
		}

		matches, scanErr := searchFile(path, re, rel, maxMatches-len(results))
		if scanErr != nil {
			return nil // skip files that can't be read
		}
		if len(matches) > 0 {
			fileSet[rel] = true
			results = append(results, matches...)
			if len(results) >= maxMatches {
				return filepath.SkipAll
			}
		}
		return nil
	})
	if err != nil {
		return tools.Fail("Search error: "+err.Error(), nil)
	}

	truncated := len(results) >= maxMatches
	output := strings.Join(results, "\n")
	if truncated {
		output += "\n...truncated"
	}

	return tools.Success(output, map[string]any{
		"match_count": len(results),
		"file_count":  len(fileSet),
	})
}

// searchFile scans a file line by line and returns matching lines in
// "filepath:lineNumber: content" format, up to limit matches.
func searchFile(path string, re *regexp.Regexp, relPath string, limit int) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var matches []string
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := scanner.Text()
		if re.MatchString(line) {
			matches = append(matches, fmt.Sprintf("%s:%d: %s", relPath, lineNo, line))
			if len(matches) >= limit {
				break
			}
		}
	}
	return matches, scanner.Err()
}

// isBinaryFile checks if a file is binary by looking for null bytes
// in the first 512 bytes.
func isBinaryFile(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()

	buf := make([]byte, 512)
	n, err := f.Read(buf)
	if n == 0 {
		return false
	}
	for i := 0; i < n; i++ {
		if buf[i] == 0 {
			return true
		}
	}
	return false
}

// expandBraces expands a simple brace pattern like "*.{ts,tsx}" into
// ["*.ts", "*.tsx"]. If there are no braces, returns the original
// pattern in a slice (or nil if empty).
func expandBraces(pattern string) []string {
	if pattern == "" {
		return nil
	}
	openIdx := strings.Index(pattern, "{")
	closeIdx := strings.Index(pattern, "}")
	if openIdx < 0 || closeIdx < 0 || closeIdx < openIdx {
		return []string{pattern}
	}
	prefix := pattern[:openIdx]
	suffix := pattern[closeIdx+1:]
	alternatives := strings.Split(pattern[openIdx+1:closeIdx], ",")
	var expanded []string
	for _, alt := range alternatives {
		expanded = append(expanded, prefix+strings.TrimSpace(alt)+suffix)
	}
	// Sort for deterministic matching order.
	sort.Strings(expanded)
	return expanded
}

var _ tools.Tool = (*GrepTool)(nil)
