package builtin

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// EditTool performs exact string replacement in a file.
type EditTool struct{}

func (t *EditTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "edit",
		Description: "Perform exact string replacement in a file. The oldString must match exactly once in the file unless replaceAll is true.",
		StatusText:  "Editing file...",
		Parameters: []tools.Param{
			{Name: "file_path", Type: "string", Description: "Relative path to the file", Required: true},
			{Name: "old_string", Type: "string", Description: "The exact text to find and replace", Required: true},
			{Name: "new_string", Type: "string", Description: "The replacement text", Required: true},
			{Name: "replace_all", Type: "boolean", Description: "Replace all occurrences", Required: false, Default: false},
		},
	}
}

func (t *EditTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	filePath, _ := call.Args["file_path"].(string)
	oldString, _ := call.Args["old_string"].(string)
	newString, _ := call.Args["new_string"].(string)
	replaceAll, _ := call.Args["replace_all"].(bool)

	if oldString == "" {
		return tools.Fail("oldString must not be empty", nil)
	}
	if oldString == newString {
		return tools.Fail("oldString and newString are identical", nil)
	}

	resolved, err := resolveSafePath(call.Ctx.WorkingDir, filePath)
	if err != nil {
		return tools.Fail("Invalid path: "+err.Error(), nil)
	}

	data, err := os.ReadFile(resolved)
	if err != nil {
		return tools.Fail("Failed to read file: "+err.Error(), nil)
	}
	content := string(data)

	count := strings.Count(content, oldString)
	if count == 0 {
		return tools.Fail("oldString not found in file content", nil)
	}
	if count > 1 && !replaceAll {
		return tools.Fail(
			fmt.Sprintf("Found %d matches for oldString. Use replace_all or provide more context to make it unique.", count),
			nil,
		)
	}

	var newContent string
	var replacements int
	if replaceAll {
		newContent = strings.ReplaceAll(content, oldString, newString)
		replacements = count
	} else {
		newContent = strings.Replace(content, oldString, newString, 1)
		replacements = 1
	}

	if err := os.WriteFile(resolved, []byte(newContent), 0o644); err != nil {
		return tools.Fail("Failed to write file: "+err.Error(), nil)
	}

	return tools.Success(
		fmt.Sprintf("Replaced %d occurrence(s) in %s", replacements, filePath),
		map[string]any{"path": resolved, "replacements": replacements},
	)
}

var _ tools.Tool = (*EditTool)(nil)
