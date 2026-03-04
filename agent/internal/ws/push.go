package ws

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	webpush "github.com/SherClockHolmes/webpush-go"
	"github.com/suryaumapathy2812/core-ai/agent/internal/config"
)

// PushSender sends Web Push notifications using VAPID.
type PushSender struct {
	vapidPublicKey  string
	vapidPrivateKey string
	vapidSubject    string // mailto: or https: URL
}

// NewPushSender creates a PushSender from centralized config.
// Returns nil if VAPID keys are not configured.
func NewPushSender(cfg config.VAPIDConfig) *PushSender {
	pub := strings.TrimSpace(cfg.PublicKey)
	priv := strings.TrimSpace(cfg.PrivateKey)
	subject := strings.TrimSpace(cfg.Subject)
	if pub == "" || priv == "" {
		log.Printf("ws push: VAPID keys not configured — web push disabled")
		return nil
	}
	if subject == "" {
		subject = "mailto:admin@aether.local"
	}
	log.Printf("ws push: VAPID configured (subject=%s)", subject)
	return &PushSender{
		vapidPublicKey:  pub,
		vapidPrivateKey: priv,
		vapidSubject:    subject,
	}
}

// VAPIDPublicKey returns the public key the client needs for PushManager.subscribe().
func (s *PushSender) VAPIDPublicKey() string {
	if s == nil {
		return ""
	}
	return s.vapidPublicKey
}

// PushSubscription matches the Web Push API subscription JSON.
type PushSubscription struct {
	Endpoint string `json:"endpoint"`
	Keys     struct {
		P256dh string `json:"p256dh"`
		Auth   string `json:"auth"`
	} `json:"keys"`
}

// PushPayload is the JSON sent inside the push message.
type PushPayload struct {
	Title string `json:"title"`
	Body  string `json:"body"`
	Icon  string `json:"icon,omitempty"`
	URL   string `json:"url,omitempty"`
	Tag   string `json:"tag,omitempty"`
}

// Send sends a push notification to a single subscription.
func (s *PushSender) Send(sub PushSubscription, payload PushPayload) error {
	if s == nil {
		return nil
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	wSub := &webpush.Subscription{
		Endpoint: sub.Endpoint,
		Keys: webpush.Keys{
			P256dh: sub.Keys.P256dh,
			Auth:   sub.Keys.Auth,
		},
	}

	resp, err := webpush.SendNotification(data, wSub, &webpush.Options{
		Subscriber:      s.vapidSubject,
		VAPIDPublicKey:  s.vapidPublicKey,
		VAPIDPrivateKey: s.vapidPrivateKey,
		TTL:             3600,
	})
	if err != nil {
		log.Printf("ws push: send failed endpoint=%s err=%v", sub.Endpoint, err)
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if resp.StatusCode >= 400 {
		log.Printf("ws push: endpoint returned status=%d endpoint=%s body=%s", resp.StatusCode, sub.Endpoint, string(body))
		return fmt.Errorf("push endpoint returned %d: %s", resp.StatusCode, string(body))
	}
	log.Printf("ws push: sent ok status=%d endpoint=%s", resp.StatusCode, sub.Endpoint)
	return nil
}

// PushResult captures the outcome of sending to a single subscription.
type PushResult struct {
	Endpoint string `json:"endpoint"`
	Success  bool   `json:"success"`
	Error    string `json:"error,omitempty"`
}

// SendToAll sends a push notification to all subscriptions for a user.
// Returns per-subscription results.
func (s *PushSender) SendToAll(subscriptions []PushSubscription, payload PushPayload) []PushResult {
	if s == nil {
		return nil
	}
	results := make([]PushResult, 0, len(subscriptions))
	for _, sub := range subscriptions {
		r := PushResult{Endpoint: sub.Endpoint}
		if err := s.Send(sub, payload); err != nil {
			r.Error = err.Error()
			log.Printf("ws push: failed to send to %s: %v", sub.Endpoint, err)
		} else {
			r.Success = true
		}
		results = append(results, r)
	}
	return results
}
