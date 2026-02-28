package builtin

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/db"
	"github.com/suryaumapathy/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy/core-ai/agent/internal/reminders"
	"github.com/suryaumapathy/core-ai/agent/internal/skills"
	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

func TestRunCommandRestricted(t *testing.T) {
	tool := &RunCommandTool{}
	ctx := tools.ExecContext{WorkingDir: t.TempDir()}

	ok := tool.Execute(context.Background(), tools.Call{Args: map[string]any{"program": "pwd", "args": []any{}, "timeout_sec": 5}, Ctx: ctx})
	if ok.Error {
		t.Fatalf("expected success: %#v", ok)
	}

	bad := tool.Execute(context.Background(), tools.Call{Args: map[string]any{"program": "rm", "args": []any{"-rf", "/"}, "timeout_sec": 5}, Ctx: ctx})
	if !bad.Error {
		t.Fatalf("expected restriction failure")
	}
}

func TestScheduleReminderTool(t *testing.T) {
	store := openStore(t)
	defer store.Close()
	tool := &ScheduleReminderTool{}
	runAt := time.Now().UTC().Add(time.Minute).Format(time.RFC3339)
	res := tool.Execute(context.Background(), tools.Call{Args: map[string]any{"message": "ping", "iso_datetime": runAt}, Ctx: tools.ExecContext{Store: store}})
	if res.Error {
		t.Fatalf("expected schedule success: %#v", res)
	}
	jobs, err := store.ListCronJobsByModule(context.Background(), reminders.CronModuleReminders)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if len(jobs) != 1 {
		t.Fatalf("expected one reminder job")
	}
}

func TestReadWriteListTools(t *testing.T) {
	root := t.TempDir()
	write := &WriteFileTool{}
	read := &ReadFileTool{}
	list := &ListDirectoryTool{}

	writeRes := write.Execute(context.Background(), tools.Call{Args: map[string]any{"path": "notes/todo.txt", "content": "a\nb\nc", "append": false}, Ctx: tools.ExecContext{WorkingDir: root}})
	if writeRes.Error {
		t.Fatalf("write failed: %#v", writeRes)
	}

	readRes := read.Execute(context.Background(), tools.Call{Args: map[string]any{"path": "notes/todo.txt", "offset": 2, "limit": 1}, Ctx: tools.ExecContext{WorkingDir: root}})
	if readRes.Error || !strings.Contains(readRes.Output, "2: b") {
		t.Fatalf("unexpected read result: %#v", readRes)
	}

	listRes := list.Execute(context.Background(), tools.Call{Args: map[string]any{"path": "notes"}, Ctx: tools.ExecContext{WorkingDir: root}})
	if listRes.Error || !strings.Contains(listRes.Output, "todo.txt") {
		t.Fatalf("unexpected list result: %#v", listRes)
	}

	bad := read.Execute(context.Background(), tools.Call{Args: map[string]any{"path": "../outside.txt"}, Ctx: tools.ExecContext{WorkingDir: root}})
	if !bad.Error {
		t.Fatalf("expected path escape to fail")
	}

	if _, err := filepath.Abs(root); err != nil {
		t.Fatalf("abs should work: %v", err)
	}
}

func TestManagementTools(t *testing.T) {
	root := t.TempDir()
	store := openStore(t)
	defer store.Close()

	skillsManager := skills.NewManager(skills.ManagerOptions{
		BuiltinDirs: []string{filepath.Join(root, "skills", "builtin")},
		UserDir:     filepath.Join(root, "skills", "user"),
		ExternalDir: filepath.Join(root, "skills", "external"),
	})
	_, _ = skillsManager.Discover(context.Background())

	pluginsManager := plugins.NewManager(plugins.ManagerOptions{
		BuiltinDirs: []string{filepath.Join(root, "plugins", "builtin")},
		UserDir:     filepath.Join(root, "plugins", "user"),
		ExternalDir: filepath.Join(root, "plugins", "external"),
		StateStore:  store,
	})

	ctx := tools.ExecContext{WorkingDir: root, Store: store, Skills: skillsManager, Plugins: pluginsManager}

	create := (&CreateSkillTool{}).Execute(context.Background(), tools.Call{Args: map[string]any{"name": "notes", "description": "note workflow", "content": "# hello"}, Ctx: ctx})
	if create.Error {
		t.Fatalf("create skill failed: %#v", create)
	}

	search := (&SearchSkillTool{}).Execute(context.Background(), tools.Call{Args: map[string]any{"query": "note"}, Ctx: ctx})
	if search.Error || !strings.Contains(search.Output, "notes") {
		t.Fatalf("search skill failed: %#v", search)
	}

	if err := store.UpsertPlugin(context.Background(), db.PluginRecord{Name: "weather", DisplayName: "Weather", Description: "Forecast"}); err != nil {
		t.Fatalf("upsert plugin: %v", err)
	}

	enable := (&EnablePluginTool{}).Execute(context.Background(), tools.Call{Args: map[string]any{"name": "weather"}, Ctx: ctx})
	if enable.Error {
		t.Fatalf("enable plugin failed: %#v", enable)
	}

	listJobs := (&ListJobsTool{}).Execute(context.Background(), tools.Call{Args: map[string]any{"module": ""}, Ctx: ctx})
	if listJobs.Error {
		t.Fatalf("list jobs failed: %#v", listJobs)
	}
}

func openStore(t *testing.T) *db.Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	s, err := db.Open(path)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return s
}
