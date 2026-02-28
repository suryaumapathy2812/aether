package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/suryaumapathy/core-ai/agent/internal/plugins"
)

type config struct {
	assetsDir   string
	builtinDirs string
	userDir     string
	externalDir string
	rawBaseURL  string
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	ctx := context.Background()
	cmd := os.Args[1]

	switch cmd {
	case "discover":
		cfg, args := parseGlobal(os.Args[2:])
		exitIfUnexpectedArgs(args)
		mgr := mustManager(cfg)
		count, err := mgr.Discover(ctx)
		exitIfErr(err)
		fmt.Printf("discovered %d plugins\n", count)

	case "list":
		cfg, args := parseGlobal(os.Args[2:])
		exitIfUnexpectedArgs(args)
		mgr := mustManager(cfg)
		discoverOrDie(ctx, mgr)
		for _, p := range mgr.List() {
			fmt.Printf("- %s [%s] %s\n", p.Name, p.Source, p.Description)
		}

	case "read":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: plugins read <name>")
		}
		mgr := mustManager(cfg)
		discoverOrDie(ctx, mgr)
		manifest, err := mgr.ReadManifest(args[0])
		exitIfErr(err)
		b, err := json.MarshalIndent(manifest, "", "  ")
		exitIfErr(err)
		fmt.Println(string(b))

	case "search":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) == 0 {
			exitWith("usage: plugins search <query>")
		}
		mgr := mustManager(cfg)
		discoverOrDie(ctx, mgr)
		query := strings.Join(args, " ")
		for _, p := range mgr.Search(query) {
			fmt.Printf("- %s [%s] %s\n", p.Name, p.Source, p.Description)
		}

	case "read-skill":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: plugins read-skill <name>")
		}
		mgr := mustManager(cfg)
		discoverOrDie(ctx, mgr)
		content, err := mgr.ReadSkill(args[0])
		exitIfErr(err)
		fmt.Println(content)

	case "remove":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: plugins remove <name>")
		}
		mgr := mustManager(cfg)
		discoverOrDie(ctx, mgr)
		exitIfErr(mgr.Remove(args[0]))
		fmt.Printf("removed %s\n", args[0])

	case "install":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: plugins install <owner/repo[@plugin-path]>")
		}
		mgr := mustManager(cfg)
		discoverOrDie(ctx, mgr)
		res, err := mgr.InstallFromSource(ctx, args[0])
		exitIfErr(err)
		fmt.Printf("installed %s from %s\n", res.Installed.Name, res.RemoteURL)

	default:
		printUsage()
		os.Exit(1)
	}
}

func parseGlobal(args []string) (config, []string) {
	fs := flag.NewFlagSet("plugins", flag.ExitOnError)
	assets := fs.String("assets-dir", defaultAssetsDir(), "assets root directory")
	builtin := fs.String("builtin-dirs", "", "comma-separated builtin plugin dirs (defaults to <assets-dir>/plugins/builtin)")
	user := fs.String("user-dir", "", "user plugin directory (defaults to <assets-dir>/plugins/user)")
	external := fs.String("external-dir", "", "external installed plugin directory (defaults to <assets-dir>/plugins/external)")
	rawBase := fs.String("raw-base-url", "https://raw.githubusercontent.com", "base URL for install source fetches")
	_ = fs.Parse(args)

	return config{
		assetsDir:   *assets,
		builtinDirs: *builtin,
		userDir:     *user,
		externalDir: *external,
		rawBaseURL:  *rawBase,
	}, fs.Args()
}

func mustManager(cfg config) *plugins.Manager {
	assetsDir := cfg.assetsDir
	if strings.TrimSpace(assetsDir) == "" {
		assetsDir = defaultAssetsDir()
	}
	builtinDefault := filepath.Join(assetsDir, "plugins", "builtin")
	userDir := cfg.userDir
	if strings.TrimSpace(userDir) == "" {
		userDir = filepath.Join(assetsDir, "plugins", "user")
	}
	externalDir := cfg.externalDir
	if strings.TrimSpace(externalDir) == "" {
		externalDir = filepath.Join(assetsDir, "plugins", "external")
	}
	builtinInput := cfg.builtinDirs
	if strings.TrimSpace(builtinInput) == "" {
		builtinInput = builtinDefault
	}
	builtinDirs := []string{}
	for _, p := range strings.Split(builtinInput, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			builtinDirs = append(builtinDirs, p)
		}
	}
	_ = os.MkdirAll(userDir, 0o755)
	_ = os.MkdirAll(externalDir, 0o755)

	return plugins.NewManager(plugins.ManagerOptions{
		BuiltinDirs: builtinDirs,
		UserDir:     userDir,
		ExternalDir: externalDir,
		RawBaseURL:  cfg.rawBaseURL,
	})
}

func discoverOrDie(ctx context.Context, mgr *plugins.Manager) {
	_, err := mgr.Discover(ctx)
	exitIfErr(err)
}

func defaultAssetsDir() string {
	wd, err := os.Getwd()
	if err != nil {
		return "assets"
	}
	return filepath.Clean(filepath.Join(wd, "assets"))
}

func exitIfUnexpectedArgs(args []string) {
	if len(args) > 0 {
		exitWith("unexpected args: " + strings.Join(args, " "))
	}
}

func exitIfErr(err error) {
	if err == nil {
		return
	}
	if errors.Is(err, plugins.ErrNotFound) {
		exitWith("not found")
	}
	if errors.Is(err, plugins.ErrProtected) {
		exitWith("plugin is protected and cannot be removed")
	}
	if errors.Is(err, plugins.ErrDuplicateName) {
		exitWith("duplicate plugin name")
	}
	exitWith(err.Error())
}

func exitWith(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}

func printUsage() {
	fmt.Println("plugins: tiny CLI for plugins module")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  discover")
	fmt.Println("  list")
	fmt.Println("  read <name>")
	fmt.Println("  search <query>")
	fmt.Println("  read-skill <name>")
	fmt.Println("  remove <name>")
	fmt.Println("  install <owner/repo[@plugin-path]>")
	fmt.Println()
	fmt.Println("Global flags (before command):")
	fmt.Println("  --assets-dir <path>")
	fmt.Println("  --builtin-dirs <comma-separated-paths>")
	fmt.Println("  --user-dir <path>")
	fmt.Println("  --external-dir <path>")
	fmt.Println("  --raw-base-url <url>")
}
