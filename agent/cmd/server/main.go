package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	adminhttp "github.com/suryaumapathy2812/core-ai/agent/internal/admin/httpapi"
	agentauth "github.com/suryaumapathy2812/core-ai/agent/internal/auth"
	"github.com/suryaumapathy2812/core-ai/agent/internal/buildinfo"
	"github.com/suryaumapathy2812/core-ai/agent/internal/channels"
	channelshttp "github.com/suryaumapathy2812/core-ai/agent/internal/channels/httpapi"
	"github.com/suryaumapathy2812/core-ai/agent/internal/channels/telegram"
	"github.com/suryaumapathy2812/core-ai/agent/internal/config"
	"github.com/suryaumapathy2812/core-ai/agent/internal/conversation"
	convhttp "github.com/suryaumapathy2812/core-ai/agent/internal/conversation/httpapi"
	convws "github.com/suryaumapathy2812/core-ai/agent/internal/conversation/wsapi"
	"github.com/suryaumapathy2812/core-ai/agent/internal/cron"
	"github.com/suryaumapathy2812/core-ai/agent/internal/dataapi"
	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
	"github.com/suryaumapathy2812/core-ai/agent/internal/llm"
	llmhttp "github.com/suryaumapathy2812/core-ai/agent/internal/llm/httpapi"
	"github.com/suryaumapathy2812/core-ai/agent/internal/media"
	"github.com/suryaumapathy2812/core-ai/agent/internal/memory"
	"github.com/suryaumapathy2812/core-ai/agent/internal/observability"
	"github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/proactive"
	"github.com/suryaumapathy2812/core-ai/agent/internal/providers"
	"github.com/suryaumapathy2812/core-ai/agent/internal/reminders"
	"github.com/suryaumapathy2812/core-ai/agent/internal/skills"
	skillshttp "github.com/suryaumapathy2812/core-ai/agent/internal/skills/httpapi"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools/builtin"
	toolhttp "github.com/suryaumapathy2812/core-ai/agent/internal/tools/httpapi"
	plugintools "github.com/suryaumapathy2812/core-ai/agent/internal/tools/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/updater"
	"github.com/suryaumapathy2812/core-ai/agent/internal/ws"
)

func main() {
	loadDotEnvIfPresent()
	observability.Init("agent")
	http.DefaultTransport = observability.WrapTransport(http.DefaultTransport)

	// ── Load & validate centralized config ──────────────────────────
	cfg := config.Load()
	if err := cfg.Validate(); err != nil {
		log.Fatalf("config validation failed:\n%v", err)
	}

	// ── Database ────────────────────────────────────────────────────
	store, err := db.OpenInAssets(cfg.AssetsDir, cfg.StateKey)
	if err != nil {
		log.Fatalf("failed to open db: %v", err)
	}
	defer store.Close()

	scheduler := cron.NewScheduler(store, cron.SchedulerOptions{
		PollInterval: time.Second,
		LeaseFor:     30 * time.Second,
		BatchSize:    25,
		JobTimeout:   60 * time.Second,
	})

	// ── Notification infrastructure ─────────────────────────────────
	wsHub := ws.NewHub()
	pushSender := ws.NewPushSender(cfg.VAPID)

	pluginRegistry := plugins.NewCronRegistry()
	plugins.RegisterDefaultTokenRotators(pluginRegistry)
	plugins.RegisterDefaultWatchRenewers(pluginRegistry)
	plugins.RegisterCronHandlers(scheduler, store, pluginRegistry)

	reminderRegistry := reminders.NewRegistry()
	reminderRegistry.Register(func(ctx context.Context, payload map[string]any) error {
		userID := strings.TrimSpace(anyToString(payload["user_id"]))
		if userID == "" {
			userID = strings.TrimSpace(anyToString(payload["target_user_id"]))
		}
		msg := strings.TrimSpace(anyToString(payload["message"]))
		if msg == "" {
			msg = "(empty reminder payload)"
		}
		if userID != "" {
			wsHub.Broadcast(userID, ws.Message{Type: "reminder", Payload: map[string]any{"message": msg, "payload": payload}})
			if pushSender != nil {
				if subs, err := store.GetPushSubscriptions(ctx, userID); err == nil && len(subs) > 0 {
					pushSubs := make([]ws.PushSubscription, 0, len(subs))
					for _, sub := range subs {
						pushSubs = append(pushSubs, ws.PushSubscription{Endpoint: sub.Endpoint, Keys: struct {
							P256dh string `json:"p256dh"`
							Auth   string `json:"auth"`
						}{P256dh: sub.KeyP256dh, Auth: sub.KeyAuth}})
					}
					pushSender.SendToAll(pushSubs, ws.PushPayload{Title: "Reminder", Body: msg, Tag: "reminder"})
				}
			}
		}
		log.Printf("reminder delivered: user=%s message=%s", userID, msg)
		return nil
	})
	reminders.RegisterCronHandlers(scheduler, reminderRegistry)

	// ── Skills & Plugins ────────────────────────────────────────────
	skillsManager := skills.NewManager(skills.ManagerOptions{
		BuiltinDirs: []string{filepath.Join(cfg.AssetsDir, "skills", "builtin")},
		UserDir:     filepath.Join(cfg.AssetsDir, "skills", "user"),
		ExternalDir: filepath.Join(cfg.AssetsDir, "skills", "external"),
	})
	if _, err := skillsManager.Discover(context.Background()); err != nil {
		log.Printf("skills discover warning: %v", err)
	}

	pluginsManager := plugins.NewManager(plugins.ManagerOptions{
		BuiltinDirs: []string{filepath.Join(cfg.AssetsDir, "plugins", "builtin")},
		UserDir:     filepath.Join(cfg.AssetsDir, "plugins", "user"),
		ExternalDir: filepath.Join(cfg.AssetsDir, "plugins", "external"),
		StateStore:  store,
	})
	if _, err := pluginsManager.Discover(context.Background()); err != nil {
		log.Printf("plugins discover warning: %v", err)
	}

	// ── Tools ───────────────────────────────────────────────────────
	toolRegistry := tools.NewRegistry()
	if err := builtin.RegisterAll(toolRegistry); err != nil {
		log.Fatalf("failed to register core tools: %v", err)
	}
	if err := plugintools.RegisterAvailable(toolRegistry, pluginsManager); err != nil {
		log.Fatalf("failed to register plugin tools: %v", err)
	}
	workspaceDir := filepath.Join(cfg.AssetsDir, "workspace")
	if err := os.MkdirAll(workspaceDir, 0o755); err != nil {
		log.Fatalf("failed to create workspace dir: %v", err)
	}
	pushDeliverer := ws.NewPushDeliverer(store, pushSender, wsHub)
	directValidator := agentauth.NewValidator(cfg.DirectTokenSecret, cfg.Channels.AgentID)

	// ── Question system & WS notify ────────────────────────────────
	// Shared WS notify callback used by both the conversation handler and
	// the question asker bridge.
	wsNotify := func(userID, eventType string, payload map[string]any) {
		wsHub.Broadcast(userID, ws.Message{Type: eventType, Payload: payload})
	}
	// Lazy wrapper: the orchestrator needs a QuestionAsker at creation time,
	// but the conversation handler (which owns the questionManager) is created
	// later. The inner implementation is wired after the handler is created.
	questionAskerHolder := &lazyQuestionAsker{}

	toolOrchestrator := tools.NewOrchestrator(toolRegistry, tools.ExecContext{
		WorkingDir:    workspaceDir,
		Store:         store,
		Skills:        skillsManager,
		Plugins:       pluginsManager,
		PushDeliverer: pushDeliverer,
		QuestionAsker: questionAskerHolder,
	})

	// ── LLM & Media ────────────────────────────────────────────────
	mux := http.NewServeMux()
	llmProvider := providers.NewOpenAILLMProvider(cfg.LLM)
	llmCore := llm.NewCore(llmProvider, toolOrchestrator)

	// Embedding provider for memory vector search
	embeddingProvider := providers.NewEmbeddingProvider(
		cfg.LLM.EmbeddingAPIKey,
		cfg.LLM.EmbeddingBaseURL,
		cfg.LLM.EmbeddingModel,
	)

	toolOrchestrator = tools.NewOrchestrator(toolRegistry, tools.ExecContext{
		WorkingDir:        workspaceDir,
		Store:             store,
		Skills:            skillsManager,
		Plugins:           pluginsManager,
		PushDeliverer:     pushDeliverer,
		QuestionAsker:     questionAskerHolder,
		EmbeddingProvider: embeddingProvider,
	})

	mediaService, err := media.New(context.Background(), cfg.S3)
	if err != nil {
		log.Fatalf("failed to init media storage: %v", err)
	}
	memoryService := memory.NewService(store, llmCore, embeddingProvider)
	llmBuilder := llm.NewContextBuilder(toolRegistry, skillsManager, pluginsManager, store, llm.ContextBuilderConfig{
		SystemPrompt: cfg.SystemPrompt,
		PromptFile:   cfg.PromptFile,
		AssetsDir:    cfg.AssetsDir,
		Embedder:     embeddingProvider,
	})

	// ── Proactive engine ────────────────────────────────────────────
	proactiveEngine := proactive.New(proactive.Options{
		Store:               store,
		Core:                llmCore,
		PushSender:          pushSender,
		WSHub:               wsHub,
		AssetsDir:           cfg.AssetsDir,
		PlanIntervalSeconds: cfg.Proactive.PlanIntervalSeconds,
	})
	proactive.RegisterCronHandlers(scheduler, proactiveEngine)

	// ── Memory dedup engine ────────────────────────────────────────
	dedupEngine := memory.NewDedupEngine(memory.DedupOptions{
		Store:        store,
		IntervalSecs: memory.DefaultDedupIntervalSecs, // 12 hours
	})
	memory.RegisterDedupCronHandlers(scheduler, dedupEngine)

	// ── HTTP handlers ───────────────────────────────────────────────
	conversationRuntime := conversation.NewRuntime(conversation.RuntimeOptions{Core: llmCore})
	llmHandler := llmhttp.New(llmhttp.Options{
		Core: llmCore, Builder: llmBuilder, Memory: memoryService, Media: mediaService,
		Model: cfg.LLM.Model, MediaLimits: cfg.Media,
	})
	llmHandler.RegisterRoutes(mux)
	convHandler := convhttp.New(convhttp.Options{Runtime: conversationRuntime, Builder: llmBuilder, Memory: memoryService, Media: mediaService, Store: store, Limits: cfg.Media, Notify: wsNotify, Validator: directValidator})
	// Wire the question asker bridge now that the handler (and its question manager) exist.
	questionAskerHolder.inner = convhttp.NewQuestionAskerBridge(convHandler.QuestionManager(), wsNotify)
	convHandler.RegisterRoutes(mux)
	convWSHandler := convws.New(convws.Options{Runtime: conversationRuntime, Builder: llmBuilder, Memory: memoryService, Store: store, Limits: cfg.Media, Notify: wsNotify, Validator: directValidator})
	convWSHandler.RegisterRoutes(mux)
	dataHandler := dataapi.New(dataapi.Options{Store: store, Media: mediaService, Embedder: embeddingProvider, Validator: directValidator})
	dataHandler.RegisterRoutes(mux)
	skillsHandler := skillshttp.New(skillshttp.Options{Manager: skillsManager, Store: store, Validator: directValidator})
	skillsHandler.RegisterRoutes(mux)

	// WebSocket + Web Push endpoints
	wsHandler := ws.NewHandler(wsHub, directValidator)
	wsHandler.RegisterRoutes(mux)
	pushHandler := ws.NewPushHandler(store, pushSender, directValidator)
	pushHandler.RegisterRoutes(mux)
	preferencesHandler := ws.NewPreferencesHandler(store, directValidator)
	preferencesHandler.RegisterRoutes(mux)

	handler := toolhttp.New(toolhttp.Options{Registry: toolRegistry, Orchestrator: toolOrchestrator, Plugins: pluginsManager, Store: store, Validator: directValidator})
	handler.RegisterRoutes(mux)
	if err := handler.EnsurePluginCronJobs(context.Background()); err != nil {
		log.Printf("plugin cron schedule warning: %v", err)
	}
	if err := proactiveEngine.EnsureCronJobs(context.Background()); err != nil {
		log.Printf("proactive cron schedule warning: %v", err)
	}
	if err := dedupEngine.EnsureCronJobs(context.Background()); err != nil {
		log.Printf("memory dedup cron schedule warning: %v", err)
	}

	up := updater.New(updater.Config{
		CurrentVersion: buildinfo.Version,
		Repo:           cfg.UpdateRepo,
		Token:          cfg.UpdateToken,
		AssetsDir:      cfg.AssetsDir,
	})
	adminHandler := adminhttp.New(adminhttp.Options{
		Updater:    up,
		Builder:    llmBuilder,
		Skills:     skillsManager,
		Plugins:    pluginsManager,
		AdminToken: cfg.AdminToken,
	})
	adminHandler.RegisterRoutes(mux)

	// ── Channels (Telegram, WhatsApp, etc.) ────────────────────────
	channelMessageHandler := func(ctx context.Context, userID, text string, metadata map[string]any) error {
		if strings.TrimSpace(text) == "" {
			return nil
		}

		log.Printf("channel message: user=%s text=%s", userID, text)

		dbChannelID, _ := metadata["channel_id_db"].(string)

		// Build the LLM request envelope with shared cross-channel context.
		messages := buildSharedConversationMessages(ctx, store, userID, text, 8)
		policy := map[string]any{}
		sessionID := strings.TrimSpace(userID)
		if sessionID == "" {
			sessionID = "default"
		}
		env := llmBuilder.Build(messages, policy, userID, sessionID)

		// Inject user context for tool execution
		runtimeCtx := tools.WithTaskRuntimeContext(ctx, tools.TaskRuntimeContext{UserID: userID})

		// Run through the conversation runtime and collect the answer
		var answerParts []string
		for ev := range conversationRuntime.Run(runtimeCtx, env, conversation.RunOptions{}) {
			switch ev.EventType {
			case conversation.EventTextDelta:
				if t, _ := ev.Payload["delta"].(string); strings.TrimSpace(t) != "" {
					answerParts = append(answerParts, t)
				}
			case conversation.EventError:
				errMsg, _ := ev.Payload["errorText"].(string)
				log.Printf("channel LLM error: user=%s err=%s", userID, errMsg)
			}
		}

		answer := strings.TrimSpace(strings.Join(answerParts, ""))
		if answer == "" {
			answer = "I'm sorry, I couldn't generate a response."
		}

		// Record conversation in memory
		if memoryService != nil {
			memoryService.RecordConversation(context.Background(), userID, sessionID, text, text, answer)
		}

		// Store inbound message
		if dbChannelID != "" {
			_, _ = store.AddChannelMessage(ctx, db.ChannelMessageRecord{
				ChannelID: dbChannelID,
				Direction: "inbound",
				Content:   text,
			})
		}

		// Send the response back via the channel
		channelType, _ := metadata["channel_type"].(string)
		chatID, _ := metadata["chat_id"].(string)
		botToken, _ := metadata["bot_token"].(string)

		if channelType == string(channels.ChannelTypeTelegram) && chatID != "" && botToken != "" {
			tgCh := telegram.NewTelegramChannel(botToken, nil)
			if err := tgCh.SendMessage(ctx, channels.OutboundMessage{
				ChannelID: chatID,
				Text:      answer,
				ParseMode: "Markdown",
			}); err != nil {
				log.Printf("channel send error: user=%s err=%v", userID, err)
				return err
			}

			// Store outbound message
			if dbChannelID != "" {
				_, _ = store.AddChannelMessage(ctx, db.ChannelMessageRecord{
					ChannelID: dbChannelID,
					Direction: "outbound",
					Content:   answer,
				})
			}
		}

		log.Printf("channel reply sent: user=%s len=%d", userID, len(answer))
		return nil
	}
	channelHandler := channelshttp.NewHandler(store, channelMessageHandler, cfg.Channels.WebhookURL, cfg.Channels.AgentID)
	channelHandler.RegisterRoutes(mux)

	// ── Start server ────────────────────────────────────────────────
	corsMiddleware := observability.CORSMiddleware(observability.CORSConfig{
		AllowedOrigins:   cfg.CORS.AllowedOrigins,
		AllowedMethods:   cfg.CORS.AllowedMethods,
		AllowedHeaders:   cfg.CORS.AllowedHeaders,
		ExposeHeaders:    cfg.CORS.ExposeHeaders,
		MaxAge:           cfg.CORS.MaxAge,
		AllowCredentials: cfg.CORS.AllowCredentials,
	})
	httpServer := &http.Server{Addr: ":" + strconv.Itoa(cfg.Port), Handler: corsMiddleware(observability.Middleware(mux))}
	go func() {
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("http server error: %v", err)
		}
	}()
	log.Printf("http server listening on %s", httpServer.Addr)
	logStartupReport(context.Background(), store, skillsManager, pluginsManager, toolRegistry, pluginRegistry, reminderRegistry)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()
	memoryService.Start(ctx)

	if err := scheduler.Start(ctx); err != nil {
		log.Fatalf("failed to start scheduler: %v", err)
	}
	log.Printf("server bootstrap started (assets=%s)", cfg.AssetsDir)

	<-ctx.Done()
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	if err := scheduler.Stop(shutdownCtx); err != nil {
		log.Printf("scheduler stop error: %v", err)
	}
	if err := memoryService.Stop(shutdownCtx); err != nil {
		log.Printf("memory service stop error: %v", err)
	}
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("http shutdown error: %v", err)
	}
	log.Printf("server shutdown complete")
}

func loadDotEnvIfPresent() {
	wd, err := os.Getwd()
	if err != nil {
		return
	}
	path := filepath.Join(wd, ".env")
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}
		eq := strings.Index(line, "=")
		if eq <= 0 {
			continue
		}
		key := strings.TrimSpace(line[:eq])
		value := strings.TrimSpace(line[eq+1:])
		if key == "" {
			continue
		}
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'') {
				value = value[1 : len(value)-1]
			}
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		_ = os.Setenv(key, value)
	}
}

// lazyQuestionAsker is a tools.QuestionAsker wrapper that delegates to an
// inner implementation once it's set. This solves the chicken-and-egg problem
// where the orchestrator (which needs a QuestionAsker) is created before the
// conversation handler (which provides the QuestionAsker implementation).
type lazyQuestionAsker struct {
	inner tools.QuestionAsker
}

func (l *lazyQuestionAsker) AskQuestion(ctx context.Context, sessionID string, question string, header string, options []map[string]any, allowCustom bool) ([]string, error) {
	if l.inner == nil {
		return nil, fmt.Errorf("question system not initialized")
	}
	return l.inner.AskQuestion(ctx, sessionID, question, header, options, allowCustom)
}

func anyToString(v any) string {
	if v == nil {
		return ""
	}
	s, ok := v.(string)
	if ok {
		return s
	}
	return strings.TrimSpace(fmt.Sprintf("%v", v))
}

func buildSharedConversationMessages(ctx context.Context, store *db.Store, userID, latestText string, maxTurns int) []map[string]any {
	latestText = strings.TrimSpace(latestText)
	if latestText == "" {
		return nil
	}
	if store == nil {
		return []map[string]any{{"role": "user", "content": latestText}}
	}
	if maxTurns <= 0 {
		maxTurns = 8
	}

	conversations, err := store.ListMemoryConversations(ctx, strings.TrimSpace(userID), maxTurns)
	if err != nil || len(conversations) == 0 {
		return []map[string]any{{"role": "user", "content": latestText}}
	}

	messages := make([]map[string]any, 0, len(conversations)*2+1)
	for i := len(conversations) - 1; i >= 0; i-- {
		c := conversations[i]
		if um := strings.TrimSpace(c.UserMessage); um != "" {
			messages = append(messages, map[string]any{"role": "user", "content": um})
		}
		if am := strings.TrimSpace(c.AssistantMessage); am != "" {
			messages = append(messages, map[string]any{"role": "assistant", "content": am})
		}
	}

	messages = append(messages, map[string]any{"role": "user", "content": latestText})
	if len(messages) > maxTurns*2+1 {
		messages = messages[len(messages)-(maxTurns*2+1):]
	}
	return messages
}

func logStartupReport(ctx context.Context, store *db.Store, skillsManager *skills.Manager, pluginsManager *plugins.Manager, toolRegistry *tools.Registry, pluginRegistry *plugins.CronRegistry, reminderRegistry *reminders.Registry) {
	skillCount := 0
	if skillsManager != nil {
		skillCount = len(skillsManager.List())
	}

	pluginCount := 0
	enabledPlugins := 0
	if pluginsManager != nil {
		pluginCount = len(pluginsManager.List())
	}
	if store != nil {
		if records, err := store.ListPlugins(ctx); err == nil {
			for _, rec := range records {
				if rec.Enabled {
					enabledPlugins++
				}
			}
		}
	}

	totalTools := 0
	builtinTools := 0
	pluginTools := 0
	if toolRegistry != nil {
		names := toolRegistry.ToolNames()
		totalTools = len(names)
		for _, name := range names {
			if toolRegistry.PluginForTool(name) == "" {
				builtinTools++
			} else {
				pluginTools++
			}
		}
	}

	cronTotal := 0
	cronScheduled := 0
	cronRetry := 0
	cronRunning := 0
	if store != nil {
		if jobs, err := store.ListCronJobs(ctx); err == nil {
			cronTotal = len(jobs)
			for _, job := range jobs {
				switch job.Status {
				case db.CronStatusScheduled:
					cronScheduled++
				case db.CronStatusRetry:
					cronRetry++
				case db.CronStatusRunning:
					cronRunning++
				}
			}
		}
	}

	rotatorCount := 0
	renewerCount := 0
	if pluginRegistry != nil {
		rotatorCount = pluginRegistry.TokenRotatorCount()
		renewerCount = pluginRegistry.WatchRenewerCount()
	}
	reminderHandler := false
	if reminderRegistry != nil {
		reminderHandler = reminderRegistry.HandlerRegistered()
	}

	log.Printf("startup report: skills=%d plugins=%d enabled_plugins=%d", skillCount, pluginCount, enabledPlugins)
	log.Printf("startup report: tools_total=%d builtin_tools=%d plugin_tools=%d", totalTools, builtinTools, pluginTools)
	log.Printf("startup report: cron_jobs_total=%d scheduled=%d retry=%d running=%d", cronTotal, cronScheduled, cronRetry, cronRunning)
	log.Printf("startup report: cron_handlers token_rotators=%d watch_renewers=%d reminder_handler=%t", rotatorCount, renewerCount, reminderHandler)
}
