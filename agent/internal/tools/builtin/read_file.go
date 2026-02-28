package builtin

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

type ReadFileTool struct{}

func (t *ReadFileTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "read_file",
		Description: "Read a text file from the working directory.",
		StatusText:  "Reading file...",
		Parameters: []tools.Param{
			{Name: "path", Type: "string", Description: "Relative file path", Required: true},
			{Name: "offset", Type: "integer", Description: "1-indexed line offset", Required: false, Default: 1},
			{Name: "limit", Type: "integer", Description: "Maximum lines to read", Required: false, Default: 200},
		},
	}
}

func (t *ReadFileTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	path, _ := call.Args["path"].(string)
	offset, _ := toolsAsInt(call.Args["offset"])
	limit, _ := toolsAsInt(call.Args["limit"])
	if offset < 1 {
		offset = 1
	}
	if limit <= 0 {
		limit = 200
	}
	resolved, err := resolveSafePath(call.Ctx.WorkingDir, path)
	if err != nil {
		return tools.Fail("Invalid path: "+err.Error(), nil)
	}
	f, err := os.Open(resolved)
	if err != nil {
		return tools.Fail("Failed to open file: "+err.Error(), nil)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	lineNo := 0
	count := 0
	lines := []string{}
	for scanner.Scan() {
		lineNo++
		if lineNo < offset {
			continue
		}
		lines = append(lines, fmt.Sprintf("%d: %s", lineNo, scanner.Text()))
		count++
		if count >= limit {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return tools.Fail("Failed reading file: "+err.Error(), nil)
	}
	return tools.Success(strings.Join(lines, "\n"), map[string]any{"path": resolved, "offset": offset, "limit": limit})
}

var _ tools.Tool = (*ReadFileTool)(nil)
