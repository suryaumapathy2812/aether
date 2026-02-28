package reminders

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/suryaumapathy2812/core-ai/agent/internal/cron"
)

const (
	CronModuleReminders = "reminders"
	CronJobTypeDeliver  = "deliver"
)

type DeliveryHandler func(ctx context.Context, payload map[string]any) error

type Registry struct {
	mu      sync.RWMutex
	handler DeliveryHandler
}

func NewRegistry() *Registry {
	return &Registry{}
}

func (r *Registry) Register(handler DeliveryHandler) {
	r.mu.Lock()
	r.handler = handler
	r.mu.Unlock()
}

func (r *Registry) get() (DeliveryHandler, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.handler == nil {
		return nil, false
	}
	return r.handler, true
}

func (r *Registry) HandlerRegistered() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.handler != nil
}

func RegisterCronHandlers(scheduler *cron.Scheduler, registry *Registry) {
	if scheduler == nil || registry == nil {
		return
	}

	scheduler.RegisterHandler(CronModuleReminders, CronJobTypeDeliver, func(ctx context.Context, job cron.Job) error {
		handler, ok := registry.get()
		if !ok {
			return fmt.Errorf("no reminder delivery handler registered")
		}
		if err := handler(ctx, job.Payload); err != nil {
			return err
		}
		log.Printf("cron/reminders: delivered reminder job_id=%s", job.ID)
		return nil
	})
}
