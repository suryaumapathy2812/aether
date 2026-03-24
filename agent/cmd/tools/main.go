package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools/builtin"
)

type cfg struct {
	assetsDir string
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	ctx := context.Background()
	command := os.Args[1]

	switch command {
	case "list":
		c, args := parseGlobal(os.Args[2:])
		expectNoArgs(args)
		_, registry, _, cleanup := bootstrap(c)
		defer cleanup()
		names := registry.ToolNames()
		for _, name := range names {
			fmt.Println(name)
		}

	case "schema":
		c, args := parseGlobal(os.Args[2:])
		expectNoArgs(args)
		_, registry, _, cleanup := bootstrap(c)
		defer cleanup()
		b, err := json.MarshalIndent(registry.OpenAISchemas(), "", "  ")
		must(err)
		fmt.Println(string(b))

	case "run":
		c, args := parseGlobal(os.Args[2:])
		name, argsJSON, callID := parseRunArgs(args)
		if strings.TrimSpace(*name) == "" {
			exit("usage: tools run --name <tool-name> [--args '{}']")
		}
		var callArgs map[string]any
		if err := json.Unmarshal([]byte(*argsJSON), &callArgs); err != nil {
			exit("invalid --args JSON")
		}

		orchestrator, _, _, cleanup := bootstrap(c)
		defer cleanup()
		res := orchestrator.Execute(ctx, *name, callArgs, *callID)
		b, err := json.MarshalIndent(res, "", "  ")
		must(err)
		fmt.Println(string(b))

	default:
		usage()
		os.Exit(1)
	}
}

func bootstrap(c cfg) (*tools.Orchestrator, *tools.Registry, *db.Store, func()) {
	assets := strings.TrimSpace(c.assetsDir)
	if assets == "" {
		assets = defaultAssetsDir()
	}
	store, err := db.OpenInAssets(assets, "")
	must(err)

	skillsManager := skills.NewManager(skills.ManagerOptions{
		BuiltinDirs: []string{filepath.Join(assets, "skills", "builtin")},
		UserDir:     filepath.Join(assets, "skills", "user"),
		ExternalDir: filepath.Join(assets, "skills", "external"),
	})
	_, _ = skillsManager.Discover(context.Background())

	pluginsManager := plugins.NewManager(plugins.ManagerOptions{
		BuiltinDirs: []string{filepath.Join(assets, "plugins", "builtin")},
		UserDir:     filepath.Join(assets, "plugins", "user"),
		ExternalDir: filepath.Join(assets, "plugins", "external"),
		StateStore:  store,
	})
	_, _ = pluginsManager.Discover(context.Background())

	workspace := filepath.Join(assets, "workspace")
	_ = os.MkdirAll(workspace, 0o755)

	registry := tools.NewRegistry()
	must(builtin.RegisterAll(registry))
	orchestrator := tools.NewOrchestrator(registry, tools.ExecContext{
		WorkingDir: workspace,
		Store:      store,
		Skills:     skillsManager,
		Plugins:    pluginsManager,
	})

	cleanup := func() { _ = store.Close() }
	return orchestrator, registry, store, cleanup
}

func parseGlobal(args []string) (cfg, []string) {
	assets := defaultAssetsDir()
	rest := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--assets-dir" && i+1 < len(args) {
			assets = args[i+1]
			i++
			continue
		}
		if strings.HasPrefix(arg, "--assets-dir=") {
			assets = strings.TrimPrefix(arg, "--assets-dir=")
			continue
		}
		rest = append(rest, arg)
	}
	return cfg{assetsDir: assets}, rest
}

func parseRunArgs(args []string) (name *string, argsJSON *string, callID *string) {
	n := ""
	a := "{}"
	c := ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--name":
			if i+1 < len(args) {
				n = args[i+1]
				i++
			}
		case "--args":
			if i+1 < len(args) {
				a = args[i+1]
				i++
			}
		case "--call-id":
			if i+1 < len(args) {
				c = args[i+1]
				i++
			}
		default:
			if strings.HasPrefix(args[i], "--name=") {
				n = strings.TrimPrefix(args[i], "--name=")
			} else if strings.HasPrefix(args[i], "--args=") {
				a = strings.TrimPrefix(args[i], "--args=")
			} else if strings.HasPrefix(args[i], "--call-id=") {
				c = strings.TrimPrefix(args[i], "--call-id=")
			}
		}
	}
	return &n, &a, &c
}

func defaultAssetsDir() string {
	wd, err := os.Getwd()
	if err != nil {
		return "assets"
	}
	return filepath.Clean(filepath.Join(wd, "assets"))
}

func expectNoArgs(args []string) {
	if len(args) > 0 {
		exit("unexpected args: " + strings.Join(args, " "))
	}
}

func must(err error) {
	if err != nil {
		exit(err.Error())
	}
}

func exit(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}

func usage() {
	fmt.Println("tools: inspect and run registered tools")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  list")
	fmt.Println("  schema")
	fmt.Println("  run --name <tool-name> [--args '{}'] [--call-id id]")
	fmt.Println()
	fmt.Println("Global flags (before command):")
	fmt.Println("  --assets-dir <path>")
}
