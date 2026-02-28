package builtin

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

var allowedPrograms = map[string]struct{}{
	"ls": {}, "pwd": {}, "go": {}, "git": {}, "python3": {}, "node": {}, "npm": {}, "bun": {}, "curl": {}, "cat": {},
}

type RunCommandTool struct{}

func (t *RunCommandTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "run_command",
		Description: "Run a restricted shell command in the working directory.",
		StatusText:  "Running command...",
		Parameters: []tools.Param{
			{Name: "program", Type: "string", Description: "Executable name", Required: true},
			{Name: "args", Type: "array", Description: "Command arguments", Required: false, Default: []any{}, Items: map[string]any{"type": "string"}},
			{Name: "timeout_sec", Type: "integer", Description: "Timeout in seconds (1-120)", Required: false, Default: 15},
		},
	}
}

func (t *RunCommandTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	program, _ := call.Args["program"].(string)
	program = strings.TrimSpace(program)
	if program == "" {
		return tools.Fail("program is required", nil)
	}
	if _, ok := allowedPrograms[program]; !ok {
		return tools.Fail("program is not allowed: "+program, map[string]any{"allowed": keys(allowedPrograms)})
	}
	if strings.ContainsAny(program, `/\\`) {
		return tools.Fail("program must be a command name without path separators", nil)
	}

	args, err := parseStringArray(call.Args["args"])
	if err != nil {
		return tools.Fail("args must be an array of strings", nil)
	}
	for _, a := range args {
		if strings.ContainsAny(a, "\n\r") {
			return tools.Fail("args cannot contain newlines", nil)
		}
	}

	timeoutSec, _ := toolsAsInt(call.Args["timeout_sec"])
	if timeoutSec <= 0 {
		timeoutSec = 15
	}
	if timeoutSec > 120 {
		timeoutSec = 120
	}

	workdir, err := resolveSafePath(call.Ctx.WorkingDir, ".")
	if err != nil {
		return tools.Fail("invalid working directory: "+err.Error(), nil)
	}

	cmdCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, program, args...)
	cmd.Dir = filepath.Clean(workdir)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()
	out := strings.TrimSpace(stdout.String())
	errOut := strings.TrimSpace(stderr.String())
	if cmdCtx.Err() == context.DeadlineExceeded {
		return tools.Fail("command timed out", map[string]any{"program": program, "args": args, "timeout_sec": timeoutSec})
	}
	if err != nil {
		msg := "command failed"
		if errOut != "" {
			msg = msg + ": " + errOut
		} else {
			msg = msg + ": " + err.Error()
		}
		return tools.Fail(msg, map[string]any{"program": program, "args": args, "stdout": out, "stderr": errOut})
	}
	if out == "" && errOut != "" {
		out = errOut
	}
	if out == "" {
		out = "command completed successfully"
	}
	return tools.Success(out, map[string]any{"program": program, "args": args, "stderr": errOut})
}

func parseStringArray(v any) ([]string, error) {
	if v == nil {
		return []string{}, nil
	}
	switch arr := v.(type) {
	case []string:
		return arr, nil
	case []any:
		out := make([]string, 0, len(arr))
		for _, item := range arr {
			s, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("array item is not string")
			}
			out = append(out, s)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("invalid args")
	}
}

func keys(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

var _ tools.Tool = (*RunCommandTool)(nil)
