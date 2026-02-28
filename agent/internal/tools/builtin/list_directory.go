package builtin

import (
	"context"
	"os"
	"sort"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type ListDirectoryTool struct{}

func (t *ListDirectoryTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "list_directory",
		Description: "List files and directories under a path within the working directory.",
		StatusText:  "Listing directory...",
		Parameters:  []tools.Param{{Name: "path", Type: "string", Description: "Relative path to list", Required: false, Default: "."}},
	}
}

func (t *ListDirectoryTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	path, _ := call.Args["path"].(string)
	resolved, err := resolveSafePath(call.Ctx.WorkingDir, path)
	if err != nil {
		return tools.Fail("Invalid path: "+err.Error(), nil)
	}
	entries, err := os.ReadDir(resolved)
	if err != nil {
		return tools.Fail("Failed to list directory: "+err.Error(), nil)
	}
	lines := make([]string, 0, len(entries))
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			name += "/"
		}
		lines = append(lines, name)
	}
	sort.Strings(lines)
	return tools.Success(strings.Join(lines, "\n"), map[string]any{"count": len(lines), "path": resolved})
}

var _ tools.Tool = (*ListDirectoryTool)(nil)
