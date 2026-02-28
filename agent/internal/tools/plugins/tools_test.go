package plugins

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	pluginmgr "github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

func TestAddAndListSubscribedFeeds(t *testing.T) {
	store := openStore(t)
	defer store.Close()
	ctx := context.Background()
	if err := store.UpsertPlugin(ctx, db.PluginRecord{Name: "rss-feeds", DisplayName: "RSS", Enabled: true}); err != nil {
		t.Fatalf("upsert plugin: %v", err)
	}
	state := pluginmgr.NewPluginState(store, "rss-feeds")
	callCtx := tools.ExecContext{Store: store, PluginState: state, PluginName: "rss-feeds"}

	add := (&AddFeedTool{}).Execute(ctx, tools.Call{Args: map[string]any{"url": "https://example.com/feed.xml"}, Ctx: callCtx})
	if add.Error {
		t.Fatalf("add feed failed: %#v", add)
	}

	list := (&ListSubscribedFeedsTool{}).Execute(ctx, tools.Call{Args: map[string]any{}, Ctx: callCtx})
	if list.Error || !strings.Contains(list.Output, "https://example.com/feed.xml") {
		t.Fatalf("list feeds failed: %#v", list)
	}
}

func TestRegisterAvailable(t *testing.T) {
	root := t.TempDir()
	assetsPlugins := filepath.Join(root, "plugins", "builtin")
	if err := copyDir(filepath.Join("..", "..", "..", "assets", "plugins", "builtin"), assetsPlugins); err != nil {
		t.Fatalf("copy plugins: %v", err)
	}
	pm := pluginmgr.NewManager(pluginmgr.ManagerOptions{BuiltinDirs: []string{assetsPlugins}})
	if _, err := pm.Discover(context.Background()); err != nil {
		t.Fatalf("discover plugins: %v", err)
	}
	r := tools.NewRegistry()
	if err := RegisterAvailable(r, pm); err != nil {
		t.Fatalf("register available: %v", err)
	}
	if _, ok := r.Get("current_weather"); !ok {
		t.Fatalf("expected weather plugin tool registered")
	}
	if _, ok := r.Get("wolfram_query"); !ok {
		t.Fatalf("expected wolfram plugin tool registered")
	}
	if _, ok := r.Get("list_unread"); !ok {
		t.Fatalf("expected gmail tool registered")
	}
	if _, ok := r.Get("upcoming_events"); !ok {
		t.Fatalf("expected google-calendar tool registered")
	}
	if _, ok := r.Get("search_contacts"); !ok {
		t.Fatalf("expected google-contacts tool registered")
	}
	if _, ok := r.Get("search_drive"); !ok {
		t.Fatalf("expected google-drive tool registered")
	}
	if _, ok := r.Get("now_playing"); !ok {
		t.Fatalf("expected spotify tool registered")
	}
	if _, ok := r.Get("telegram_send_message"); !ok {
		t.Fatalf("expected telegram tool registered")
	}
	if _, ok := r.Get("make_phone_call"); !ok {
		t.Fatalf("expected vobiz tool registered")
	}
}

func openStore(t *testing.T) *db.Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state.db")
	store, err := db.Open(path)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return store
}

func copyDir(src, dst string) error {
	entries, err := filepath.Glob(filepath.Join(src, "*"))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	for _, p := range entries {
		name := filepath.Base(p)
		target := filepath.Join(dst, name)
		info, err := os.Stat(p)
		if err != nil {
			return err
		}
		if info.IsDir() {
			if err := copyDir(p, target); err != nil {
				return err
			}
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		if err := os.WriteFile(target, b, 0o644); err != nil {
			return err
		}
	}
	return nil
}
