package builtin

import (
	"context"
	"os"
	"path/filepath"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

type WriteFileTool struct{}

func (t *WriteFileTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "write_file",
		Description: "Write text content to a file in the working directory.",
		StatusText:  "Writing file...",
		Parameters: []tools.Param{
			{Name: "path", Type: "string", Description: "Relative file path", Required: true},
			{Name: "content", Type: "string", Description: "File content", Required: true},
			{Name: "append", Type: "boolean", Description: "Append instead of overwrite", Required: false, Default: false},
		},
	}
}

func (t *WriteFileTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	path, _ := call.Args["path"].(string)
	content, _ := call.Args["content"].(string)
	appendMode, _ := call.Args["append"].(bool)

	resolved, err := resolveSafePath(call.Ctx.WorkingDir, path)
	if err != nil {
		return tools.Fail("Invalid path: "+err.Error(), nil)
	}
	if err := os.MkdirAll(filepath.Dir(resolved), 0o755); err != nil {
		return tools.Fail("Failed to prepare parent directory: "+err.Error(), nil)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if appendMode {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(resolved, flags, 0o644)
	if err != nil {
		return tools.Fail("Failed to open file: "+err.Error(), nil)
	}
	defer f.Close()
	if _, err := f.WriteString(content); err != nil {
		return tools.Fail("Failed to write file: "+err.Error(), nil)
	}
	return tools.Success("File written successfully.", map[string]any{"path": resolved, "bytes": len(content), "append": appendMode})
}

var _ tools.Tool = (*WriteFileTool)(nil)
