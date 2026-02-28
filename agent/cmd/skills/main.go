package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
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
		manager := mustManager(cfg)
		count, err := manager.Discover(ctx)
		exitIfErr(err)
		fmt.Printf("discovered %d skills\n", count)

	case "list":
		cfg, args := parseGlobal(os.Args[2:])
		exitIfUnexpectedArgs(args)
		manager := mustManager(cfg)
		discoverOrDie(ctx, manager)
		for _, skill := range manager.List() {
			fmt.Printf("- %s [%s] %s\n", skill.Name, skill.Source, skill.Description)
		}

	case "read":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: skills read <name>")
		}
		manager := mustManager(cfg)
		discoverOrDie(ctx, manager)
		content, err := manager.Read(args[0])
		exitIfErr(err)
		fmt.Println(content)

	case "search":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) == 0 {
			exitWith("usage: skills search <query>")
		}
		manager := mustManager(cfg)
		discoverOrDie(ctx, manager)
		query := strings.Join(args, " ")
		matches := manager.Search(query)
		for _, skill := range matches {
			fmt.Printf("- %s [%s] %s\n", skill.Name, skill.Source, skill.Description)
		}

	case "create":
		cfg, args := parseGlobal(os.Args[2:])
		fs := flag.NewFlagSet("create", flag.ExitOnError)
		name := fs.String("name", "", "skill name")
		description := fs.String("description", "", "skill description")
		content := fs.String("content", "", "skill markdown body")
		contentFile := fs.String("content-file", "", "path to file containing skill body")
		_ = fs.Parse(args)

		if *name == "" || *description == "" {
			exitWith("usage: skills create --name <name> --description <desc> (--content <text> | --content-file <path>)")
		}

		body := *content
		if *contentFile != "" {
			b, err := os.ReadFile(*contentFile)
			exitIfErr(err)
			body = string(b)
		}
		if strings.TrimSpace(body) == "" {
			exitWith("content is required via --content or --content-file")
		}

		manager := mustManager(cfg)
		discoverOrDie(ctx, manager)
		created, err := manager.Create(*name, *description, body)
		exitIfErr(err)
		fmt.Printf("created %s at %s\n", created.Name, created.Location)

	case "remove":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: skills remove <name>")
		}
		manager := mustManager(cfg)
		discoverOrDie(ctx, manager)
		err := manager.Remove(args[0])
		exitIfErr(err)
		fmt.Printf("removed %s\n", args[0])

	case "install":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: skills install <owner/repo[@skill-name]>")
		}
		manager := mustManager(cfg)
		discoverOrDie(ctx, manager)
		result, err := manager.InstallFromSource(ctx, args[0])
		exitIfErr(err)
		fmt.Printf("installed %s from %s\n", result.Installed.Name, result.RemoteURL)

	default:
		printUsage()
		os.Exit(1)
	}
}

func parseGlobal(args []string) (config, []string) {
	fs := flag.NewFlagSet("skills", flag.ExitOnError)
	assets := fs.String("assets-dir", defaultAssetsDir(), "assets root directory")
	builtin := fs.String("builtin-dirs", "", "comma-separated builtin skill dirs (defaults to <assets-dir>/skills/builtin)")
	user := fs.String("user-dir", "", "user skills directory (defaults to <assets-dir>/skills/user)")
	external := fs.String("external-dir", "", "external installed skills directory (defaults to <assets-dir>/skills/external)")
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

func mustManager(cfg config) *skills.Manager {
	assetsDir := cfg.assetsDir
	if strings.TrimSpace(assetsDir) == "" {
		assetsDir = defaultAssetsDir()
	}

	builtinDirDefault := filepath.Join(assetsDir, "skills", "builtin")
	userDir := cfg.userDir
	if strings.TrimSpace(userDir) == "" {
		userDir = filepath.Join(assetsDir, "skills", "user")
	}
	externalDir := cfg.externalDir
	if strings.TrimSpace(externalDir) == "" {
		externalDir = filepath.Join(assetsDir, "skills", "external")
	}

	builtinDirs := []string{}
	builtinInput := cfg.builtinDirs
	if strings.TrimSpace(builtinInput) == "" {
		builtinInput = builtinDirDefault
	}
	for _, part := range strings.Split(builtinInput, ",") {
		cleaned := strings.TrimSpace(part)
		if cleaned != "" {
			builtinDirs = append(builtinDirs, cleaned)
		}
	}

	_ = os.MkdirAll(userDir, 0o755)
	_ = os.MkdirAll(externalDir, 0o755)

	return skills.NewManager(skills.ManagerOptions{
		BuiltinDirs: builtinDirs,
		UserDir:     userDir,
		ExternalDir: externalDir,
		RawBaseURL:  cfg.rawBaseURL,
	})
}

func discoverOrDie(ctx context.Context, manager *skills.Manager) {
	_, err := manager.Discover(ctx)
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
	if errors.Is(err, skills.ErrNotFound) {
		exitWith("not found")
	}
	if errors.Is(err, skills.ErrProtected) {
		exitWith("skill is protected and cannot be removed")
	}
	if errors.Is(err, skills.ErrDuplicateName) {
		exitWith("duplicate skill name")
	}
	exitWith(err.Error())
}

func exitWith(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}

func printUsage() {
	fmt.Println("skills: tiny CLI for skills module")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  discover")
	fmt.Println("  list")
	fmt.Println("  read <name>")
	fmt.Println("  search <query>")
	fmt.Println("  create --name <name> --description <desc> (--content <text> | --content-file <path>)")
	fmt.Println("  remove <name>")
	fmt.Println("  install <owner/repo[@skill-name]>")
	fmt.Println()
	fmt.Println("Global flags (before command):")
	fmt.Println("  --assets-dir <path>")
	fmt.Println("  --builtin-dirs <comma-separated-paths>")
	fmt.Println("  --user-dir <path>")
	fmt.Println("  --external-dir <path>")
	fmt.Println("  --raw-base-url <url>")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  go run ./cmd/skills list")
	fmt.Println("  go run ./cmd/skills search tool calling")
	fmt.Println("  go run ./cmd/skills read soul")
	fmt.Println("  go run ./cmd/skills create --name notes --description 'notes workflow' --content '## Steps'")
	fmt.Println("  go run ./cmd/skills install vercel-labs/agent-skills@nextjs")
}
