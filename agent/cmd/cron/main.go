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
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

type config struct {
	assetsDir string
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	ctx := context.Background()
	cmd := os.Args[1]

	switch cmd {
	case "list":
		cfg, args := parseGlobal(os.Args[2:])
		fs := flag.NewFlagSet("list", flag.ExitOnError)
		module := fs.String("module", "", "optional module filter")
		_ = fs.Parse(args)
		store := mustStore(cfg)
		defer store.Close()
		jobs := []db.CronJobRecord{}
		var err error
		if strings.TrimSpace(*module) == "" {
			jobs, err = store.ListCronJobs(ctx)
		} else {
			jobs, err = store.ListCronJobsByModule(ctx, *module)
		}
		exitIfErr(err)
		if len(jobs) == 0 {
			fmt.Println("no cron jobs found")
			return
		}
		for _, j := range jobs {
			interval := "one-shot"
			if j.IntervalS != nil {
				interval = fmt.Sprintf("every %ds", *j.IntervalS)
			}
			fmt.Printf("- %s [%s/%s] status=%s next=%s attempts=%d/%d %s\n", j.ID, j.Module, j.JobType, j.Status, j.NextRunAt.Format(time.RFC3339), j.AttemptCount, j.MaxAttempts, interval)
		}

	case "get":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: cron get <job-id>")
		}
		store := mustStore(cfg)
		defer store.Close()
		job, err := store.GetCronJob(ctx, args[0])
		exitIfErr(err)
		b, err := json.MarshalIndent(job, "", "  ")
		exitIfErr(err)
		fmt.Println(string(b))

	case "schedule":
		cfg, args := parseGlobal(os.Args[2:])
		fs := flag.NewFlagSet("schedule", flag.ExitOnError)
		module := fs.String("module", "", "module name (required)")
		jobType := fs.String("type", "", "job type (required)")
		runAtRaw := fs.String("run-at", "", "run at time in RFC3339 (required)")
		payloadRaw := fs.String("payload", "{}", "json payload")
		intervalS := fs.Int64("interval-s", 0, "recurrence interval in seconds (0 = one-shot)")
		maxAttempts := fs.Int("max-attempts", 5, "maximum attempts before failed")
		_ = fs.Parse(args)

		if strings.TrimSpace(*module) == "" || strings.TrimSpace(*jobType) == "" || strings.TrimSpace(*runAtRaw) == "" {
			exitWith("usage: cron schedule --module <module> --type <job-type> --run-at <RFC3339> [--payload '{}'] [--interval-s N] [--max-attempts N]")
		}

		runAt, err := time.Parse(time.RFC3339, *runAtRaw)
		if err != nil {
			exitWith("invalid --run-at; expected RFC3339, e.g. 2026-02-27T10:30:00Z")
		}

		var payload any = map[string]any{}
		if strings.TrimSpace(*payloadRaw) != "" {
			var parsed any
			if err := json.Unmarshal([]byte(*payloadRaw), &parsed); err != nil {
				exitWith("invalid --payload json")
			}
			payload = parsed
		}

		store := mustStore(cfg)
		defer store.Close()

		create := db.CronJobCreate{
			Module:      *module,
			JobType:     *jobType,
			Payload:     payload,
			RunAt:       runAt.UTC(),
			MaxAttempts: *maxAttempts,
		}
		if *intervalS > 0 {
			create.IntervalS = intervalS
		}

		job, err := store.ScheduleCronJob(ctx, create)
		exitIfErr(err)
		fmt.Printf("scheduled %s [%s/%s] next=%s\n", job.ID, job.Module, job.JobType, job.NextRunAt.Format(time.RFC3339))

	case "cancel":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: cron cancel <job-id>")
		}
		store := mustStore(cfg)
		defer store.Close()
		exitIfErr(store.CancelCronJob(ctx, args[0]))
		fmt.Printf("cancelled %s\n", args[0])

	case "pause":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: cron pause <job-id>")
		}
		store := mustStore(cfg)
		defer store.Close()
		exitIfErr(store.PauseCronJob(ctx, args[0]))
		fmt.Printf("paused %s\n", args[0])

	case "resume":
		cfg, args := parseGlobal(os.Args[2:])
		if len(args) != 1 {
			exitWith("usage: cron resume <job-id>")
		}
		store := mustStore(cfg)
		defer store.Close()
		exitIfErr(store.ResumeCronJob(ctx, args[0]))
		fmt.Printf("resumed %s\n", args[0])

	default:
		printUsage()
		os.Exit(1)
	}
}

func parseGlobal(args []string) (config, []string) {
	fs := flag.NewFlagSet("cron", flag.ExitOnError)
	assets := fs.String("assets-dir", defaultAssetsDir(), "assets root directory")
	_ = fs.Parse(args)
	return config{assetsDir: *assets}, fs.Args()
}

func mustStore(cfg config) *db.Store {
	assetsDir := strings.TrimSpace(cfg.assetsDir)
	if assetsDir == "" {
		assetsDir = defaultAssetsDir()
	}
	store, err := db.OpenInAssets(assetsDir, "")
	if err != nil {
		exitWith(err.Error())
	}
	return store
}

func defaultAssetsDir() string {
	wd, err := os.Getwd()
	if err != nil {
		return "assets"
	}
	return filepath.Clean(filepath.Join(wd, "assets"))
}

func exitIfErr(err error) {
	if err == nil {
		return
	}
	if errors.Is(err, db.ErrNotFound) {
		exitWith("not found")
	}
	exitWith(err.Error())
}

func exitWith(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}

func printUsage() {
	fmt.Println("cron: persistent cron jobs CLI")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  list [--module <module>]")
	fmt.Println("  get <job-id>")
	fmt.Println("  schedule --module <module> --type <job-type> --run-at <RFC3339> [--payload '{}'] [--interval-s N] [--max-attempts N]")
	fmt.Println("  cancel <job-id>")
	fmt.Println("  pause <job-id>")
	fmt.Println("  resume <job-id>")
	fmt.Println()
	fmt.Println("Global flags (before command):")
	fmt.Println("  --assets-dir <path>")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  go run ./cmd/cron list")
	fmt.Println("  go run ./cmd/cron list --module plugins")
	fmt.Println("  go run ./cmd/cron schedule --module reminders --type deliver --run-at 2026-03-01T10:00:00Z --payload '{\"message\":\"standup\"}'")
	fmt.Println("  go run ./cmd/cron schedule --module plugins --type rotate_token --run-at 2026-03-01T10:00:00Z --interval-s 3600 --payload '{\"plugin\":\"gmail\"}'")
}
