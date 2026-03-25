package builtin

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// programPolicy defines argument-level restrictions for an allowed program.
type programPolicy struct {
	// deniedFlags are flags that must not appear as arguments (e.g., "-e", "--eval").
	// Matching is exact against each argument.
	deniedFlags []string

	// deniedArgs are argument values or prefixes that are denied.
	// An argument is blocked if it exactly equals or starts with any entry.
	deniedArgs []string

	// allowedSubcommands, if non-empty, restricts the first non-flag argument
	// to one of these values. Any subcommand not in this list is denied.
	allowedSubcommands []string
}

// programPolicies maps each allowed program to its argument restrictions.
// Programs not in this map are denied entirely.
var programPolicies = map[string]programPolicy{
	// Safe utilities — no restrictions needed.
	"ls":   {},
	"pwd":  {},
	"cat":  {},
	"make": {},
	"grep": {},
	"wc":   {},
	"sort": {},
	"head": {},
	"tail": {},
	"diff": {},

	// go: allow build/test/vet/fmt/mod subcommands only.
	"go": {
		allowedSubcommands: []string{"build", "test", "vet", "fmt", "mod", "version", "env", "list", "doc", "generate"},
	},

	// git: allow most operations but deny dangerous ones.
	"git": {
		deniedArgs: []string{"push --force", "remote add", "config --global"},
		deniedFlags: []string{
			"--force",
		},
	},

	// python3: block inline code execution; allow running .py files.
	"python3": {
		deniedFlags: []string{"-c", "--command"},
	},

	// pip3: allow installing, listing, inspecting packages.
	"pip3": {
		allowedSubcommands: []string{"install", "list", "show", "freeze"},
	},

	// node: block inline code execution; allow running .js files.
	"node": {
		deniedFlags: []string{"-e", "--eval", "-p", "--print", "--input-type"},
	},

	// npm: only allow safe subcommands.
	"npm": {
		allowedSubcommands: []string{"install", "test", "ls", "ci", "audit", "outdated", "list", "view", "info", "pack", "cache", "config"},
	},

	// bun: only allow safe subcommands.
	"bun": {
		allowedSubcommands: []string{"install", "test", "pm", "add", "remove"},
	},

	// curl: only allow safe GET requests — deny data sending, output, and mutating methods.
	"curl": {
		deniedFlags: []string{
			"-d", "--data", "--data-raw", "--data-binary", "--data-urlencode",
			"-F", "--form",
			"-o", "--output", "-O",
			"-T", "--upload-file",
		},
		deniedArgs: []string{
			"-X POST", "-X PUT", "-X DELETE", "-X PATCH",
			"--request POST", "--request PUT", "--request DELETE", "--request PATCH",
		},
	},

	// find: allow file searching but deny execution and deletion.
	"find": {
		deniedFlags: []string{"-exec", "-execdir", "-delete"},
	},
}

// allowedProgramNames returns the list of allowed program names for error messages.
func allowedProgramNames() []string {
	out := make([]string, 0, len(programPolicies))
	for k := range programPolicies {
		out = append(out, k)
	}
	return out
}

// validateProgramArgs checks the arguments against the program's policy.
// Returns an error describing the first violation, or nil if all args are allowed.
func validateProgramArgs(program string, args []string) error {
	policy, ok := programPolicies[program]
	if !ok {
		return fmt.Errorf("program not allowed: %s", program)
	}

	// Check denied flags: each argument is compared exactly against denied flags.
	for _, arg := range args {
		for _, denied := range policy.deniedFlags {
			if arg == denied {
				return fmt.Errorf("argument %q is not allowed for %s", arg, program)
			}
		}
	}

	// Check denied args: match against the full argument list joined as a string,
	// and also check individual arguments for prefix matches.
	fullArgs := strings.Join(args, " ")
	for _, denied := range policy.deniedArgs {
		if strings.Contains(fullArgs, denied) {
			return fmt.Errorf("argument pattern %q is not allowed for %s", denied, program)
		}
	}

	// Check allowed subcommands: if the policy restricts subcommands, the first
	// non-flag argument must be in the allowed list.
	if len(policy.allowedSubcommands) > 0 {
		sub := firstNonFlagArg(args)
		if sub == "" {
			// No subcommand provided — allow bare invocation (e.g., "go" with no args).
			return nil
		}
		for _, allowed := range policy.allowedSubcommands {
			if sub == allowed {
				return nil
			}
		}
		return fmt.Errorf("subcommand %q is not allowed for %s (allowed: %s)",
			sub, program, strings.Join(policy.allowedSubcommands, ", "))
	}

	return nil
}

// firstNonFlagArg returns the first argument that does not start with "-".
func firstNonFlagArg(args []string) string {
	for _, a := range args {
		if !strings.HasPrefix(a, "-") {
			return a
		}
	}
	return ""
}

type RunCommandTool struct{}

func (t *RunCommandTool) Definition() tools.Definition {
	return tools.Definition{
		Name: "run_command",
		Description: "Run a restricted shell command in the working directory. " +
			"Allowed programs: ls, pwd, cat, go, git, python3, node, npm, bun, curl, " +
			"make, grep, find, wc, sort, head, tail, diff. " +
			"Some programs have argument restrictions to prevent arbitrary code execution or data exfiltration.",
		StatusText: "Running command...",
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
	if _, ok := programPolicies[program]; !ok {
		return tools.Fail("program is not allowed: "+program, map[string]any{"allowed": allowedProgramNames()})
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

	// Validate arguments against the program's policy before execution.
	if err := validateProgramArgs(program, args); err != nil {
		return tools.Fail(err.Error(), map[string]any{"program": program, "args": args})
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

var _ tools.Tool = (*RunCommandTool)(nil)
