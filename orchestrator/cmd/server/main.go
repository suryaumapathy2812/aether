package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/agent"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/config"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/schema"
	"github.com/suryaumapathy2812/core-ai/orchestrator/internal/server"
)

func main() {
	_ = godotenv.Load()
	cfg := config.Load()

	pool, err := pgxpool.New(context.Background(), cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("failed to open postgres pool: %v", err)
	}
	defer pool.Close()

	if err := schema.Bootstrap(context.Background(), pool); err != nil {
		log.Fatalf("failed to bootstrap schema: %v", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var mgr *agent.Manager
	if cfg.LocalAgentURL == "" {
		mgr, err = agent.NewManager(ctx, pool, agent.ManagerConfig{
			Image:         cfg.AgentImage,
			Network:       cfg.AgentNetwork,
			IdleTimeout:   cfg.AgentIdleTimeout,
			AgentPort:     cfg.AgentPort,
			HealthTimeout: cfg.AgentHealthTimeout,
			AdminToken:    cfg.AgentSecret,
			OpenAIAPIKey:  cfg.AgentOpenAIAPIKey,
			OpenAIBaseURL: cfg.AgentOpenAIBaseURL,
			OpenAIModel:   cfg.AgentOpenAIModel,
			AgentStateKey: cfg.AgentStateKey,
			VapidPublic:   cfg.VapidPublicKey,
			VapidPrivate:  cfg.VapidPrivateKey,
			VapidSubject:  cfg.VapidSubject,
			S3Bucket:      cfg.S3Bucket,
			S3Template:    cfg.S3BucketTemplate,
			S3Region:      cfg.S3Region,
			S3AccessKey:   cfg.S3AccessKeyID,
			S3SecretKey:   cfg.S3SecretAccessKey,
			S3Endpoint:    cfg.S3Endpoint,
			S3PublicBase:  cfg.S3PublicBaseURL,
			S3ForcePath:   cfg.S3ForcePathStyle,
			S3PutTTL:      cfg.S3PutURLTTLSeconds,
			S3GetTTL:      cfg.S3GetURLTTLSeconds,
			UpdateRepo:    cfg.AgentUpdateRepo,
			UpdateToken:   cfg.AgentUpdateToken,
		})
		if err != nil {
			log.Printf("agent manager disabled: %v", err)
		} else {
			go mgr.RunIdleReaper(ctx)
		}
	}

	s := server.New(cfg, pool, mgr)
	httpServer := &http.Server{
		Addr:    ":" + strconv.Itoa(cfg.Port),
		Handler: s.Handler(),
	}

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownCtx)
		if mgr != nil {
			_ = mgr.Close()
		}
	}()

	log.Printf("orchestrator listening on %s", httpServer.Addr)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server stopped: %v", err)
	}
}
