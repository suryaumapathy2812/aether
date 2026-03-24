package builtin

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

// credentialEnvMapping maps plugin names to the environment variable names
// their access tokens will be injected as.
var credentialEnvMapping = map[string]string{
	"gmail":           "GMAIL_ACCESS_TOKEN",
	"google-calendar": "GOOGLE_CALENDAR_ACCESS_TOKEN",
	"google-contacts": "GOOGLE_CONTACTS_ACCESS_TOKEN",
	"google-drive":    "GOOGLE_DRIVE_ACCESS_TOKEN",
	"spotify":         "SPOTIFY_ACCESS_TOKEN",
	"weather":         "WEATHER_API_KEY",
	"brave-search":    "BRAVE_SEARCH_API_KEY",
	"wolfram":         "WOLFRAM_APP_ID",
}

// envVarForPlugin returns the environment variable name for a plugin's credential.
// Falls back to PLUGINNAME_ACCESS_TOKEN for unknown plugins.
func envVarForPlugin(pluginName string) string {
	if env, ok := credentialEnvMapping[pluginName]; ok && env != "" {
		return env
	}
	upper := strings.ToUpper(strings.ReplaceAll(pluginName, "-", "_"))
	return upper + "_ACCESS_TOKEN"
}

type ExecuteTool struct{}

func (t *ExecuteTool) Definition() tools.Definition {
	return tools.Definition{
		Name: "execute",
		Description: "Execute a shell command (curl, python3, bash, etc.) in the workspace. " +
			"API credentials are injected as environment variables when you specify credential plugin names. " +
			"Write commands to interact with APIs — check skills for API documentation.",
		StatusText: "Executing...",
		Parameters: []tools.Param{
			{Name: "command", Type: "string", Description: "Shell command to execute (e.g., a curl or python3 command)", Required: true},
			{Name: "credentials", Type: "array", Description: "Plugin names whose tokens to inject as env vars (e.g., [\"gmail\", \"spotify\"])",
				Required: false, Default: []any{}, Items: map[string]any{"type": "string"}},
			{Name: "timeout_sec", Type: "integer", Description: "Timeout in seconds (1-120)", Required: false, Default: 30},
		},
	}
}

func (t *ExecuteTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	command, _ := call.Args["command"].(string)
	command = strings.TrimSpace(command)
	if command == "" {
		return tools.Fail("command is required", nil)
	}

	credNames, err := parseStringArray(call.Args["credentials"])
	if err != nil {
		return tools.Fail("credentials must be an array of strings", nil)
	}

	timeoutSec, _ := toolsAsInt(call.Args["timeout_sec"])
	if timeoutSec <= 0 {
		timeoutSec = 30
	}
	if timeoutSec > 120 {
		timeoutSec = 120
	}

	// Resolve credentials from plugin store and build env vars.
	envVars, credMeta, credErr := t.resolveCredentials(ctx, call, credNames)
	if credErr != nil {
		return tools.Fail(credErr.Error(), map[string]any{"credentials": credNames})
	}

	// Build the execution environment: inherit host env + inject plugin credentials.
	cmdEnv := os.Environ()
	cmdEnv = append(cmdEnv, envVars...)

	workdir, err := resolveSafePath(call.Ctx.WorkingDir, ".")
	if err != nil {
		return tools.Fail("invalid working directory: "+err.Error(), nil)
	}

	cmdCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "bash", "-c", command)
	cmd.Dir = filepath.Clean(workdir)
	cmd.Env = cmdEnv
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()
	out := strings.TrimSpace(stdout.String())
	errOut := strings.TrimSpace(stderr.String())

	if cmdCtx.Err() == context.DeadlineExceeded {
		return tools.Fail("command timed out", map[string]any{"timeout_sec": timeoutSec, "command": command})
	}
	if err != nil {
		msg := "command failed"
		if errOut != "" {
			msg = msg + ": " + errOut
		} else {
			msg = msg + ": " + err.Error()
		}
		return tools.Fail(msg, map[string]any{"command": command, "stdout": out, "stderr": errOut, "injected_credentials": credMeta})
	}
	if out == "" && errOut != "" {
		out = errOut
	}
	if out == "" {
		out = "command completed successfully"
	}
	return tools.Success(out, map[string]any{"command": command, "stderr": errOut, "injected_credentials": credMeta})
}

// resolveCredentials looks up each plugin's access_token from the store,
// decrypts it, and returns environment variable assignments.
func (t *ExecuteTool) resolveCredentials(ctx context.Context, call tools.Call, pluginNames []string) ([]string, map[string]string, error) {
	if len(pluginNames) == 0 || call.Ctx.Store == nil {
		return nil, nil, nil
	}

	var envVars []string
	injected := map[string]string{}

	for _, name := range pluginNames {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}

		rec, err := call.Ctx.Store.GetPlugin(ctx, name)
		if err != nil {
			return nil, nil, fmt.Errorf("plugin %q not found: %v", name, err)
		}
		if !rec.Enabled {
			return nil, nil, fmt.Errorf("plugin %q is not enabled", name)
		}
		if rec.Config == nil {
			rec.Config = map[string]string{}
		}

		// Try access_token first, then api_key, then token.
		token := ""
		for _, key := range []string{"access_token", "api_key", "token"} {
			if v := strings.TrimSpace(rec.Config[key]); v != "" {
				token = v
				break
			}
		}
		if token == "" {
			return nil, nil, fmt.Errorf("plugin %q has no access_token or api_key configured", name)
		}

		// Decrypt if encrypted.
		if strings.HasPrefix(token, "enc:v1:") {
			scope := call.Ctx.Store.ScopePlugin(name)
			decrypted, err := scope.DecryptString(token)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to decrypt credential for plugin %q: %v", name, err)
			}
			token = decrypted
		}

		envName := envVarForPlugin(name)
		envVars = append(envVars, envName+"="+token)
		injected[name] = envName
	}

	return envVars, injected, nil
}

var _ tools.Tool = (*ExecuteTool)(nil)
