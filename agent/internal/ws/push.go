package ws

import (
	"encoding/json"
	"log"
	"os"
	"strings"

	webpush "github.com/SherClockHolmes/webpush-go"
)

// PushSender sends Web Push notifications using VAPID.
type PushSender struct {
	vapidPublicKey  string
	vapidPrivateKey string
	vapidSubject    string // mailto: or https: URL
}

// NewPushSenderFromEnv creates a PushSender from environment variables.
// Returns nil if VAPID keys are not configured.
func NewPushSenderFromEnv() *PushSender {
	pub := strings.TrimSpace(os.Getenv("VAPID_PUBLIC_KEY"))
	priv := strings.TrimSpace(os.Getenv("VAPID_PRIVATE_KEY"))
	subject := strings.TrimSpace(os.Getenv("VAPID_SUBJECT"))
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
	_ = resp.Body.Close()
	if resp.StatusCode >= 400 {
		log.Printf("ws push: endpoint returned status=%d endpoint=%s", resp.StatusCode, sub.Endpoint)
	}
	return nil
}

// SendToAll sends a push notification to all subscriptions for a user.
func (s *PushSender) SendToAll(subscriptions []PushSubscription, payload PushPayload) {
	if s == nil {
		return
	}
	for _, sub := range subscriptions {
		if err := s.Send(sub, payload); err != nil {
			log.Printf("ws push: failed to send to %s: %v", sub.Endpoint, err)
		}
	}
}
