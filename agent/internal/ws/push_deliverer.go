package ws

import (
	"context"
	"log"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

// PushDelivererImpl implements tools.PushDeliverer by looking up
// subscriptions from the store and sending via PushSender.
type PushDelivererImpl struct {
	store  *db.Store
	sender *PushSender
	hub    *Hub
}

// NewPushDeliverer creates a PushDeliverer that sends Web Push
// notifications and broadcasts via WebSocket.
func NewPushDeliverer(store *db.Store, sender *PushSender, hub *Hub) *PushDelivererImpl {
	return &PushDelivererImpl{store: store, sender: sender, hub: hub}
}

// DeliverPush sends a push notification to all subscriptions for a user
// and broadcasts via WebSocket. Returns counts of sent/failed deliveries.
func (d *PushDelivererImpl) DeliverPush(ctx context.Context, userID, title, body, tag string) (sent int, failed int, err error) {
	// Broadcast via WebSocket (real-time, if user is connected)
	if d.hub != nil {
		d.hub.Broadcast(userID, Message{
			Type: "notification",
			Payload: map[string]any{
				"title": title,
				"body":  body,
				"tag":   tag,
			},
		})
	}

	// Send via Web Push (works even when browser is closed)
	if d.sender == nil {
		log.Printf("ws push deliverer: push sender not configured, skipping web push")
		return 0, 0, nil
	}
	if d.store == nil {
		log.Printf("ws push deliverer: store not available, skipping web push")
		return 0, 0, nil
	}

	subs, err := d.store.GetPushSubscriptions(ctx, userID)
	if err != nil {
		log.Printf("ws push deliverer: failed to get subscriptions for user=%s: %v", userID, err)
		return 0, 0, err
	}
	if len(subs) == 0 {
		log.Printf("ws push deliverer: no subscriptions for user=%s", userID)
		return 0, 0, nil
	}

	pushSubs := make([]PushSubscription, 0, len(subs))
	for _, sub := range subs {
		pushSubs = append(pushSubs, PushSubscription{
			Endpoint: sub.Endpoint,
			Keys: struct {
				P256dh string `json:"p256dh"`
				Auth   string `json:"auth"`
			}{P256dh: sub.KeyP256dh, Auth: sub.KeyAuth},
		})
	}

	results := d.sender.SendToAll(pushSubs, PushPayload{
		Title: title,
		Body:  body,
		Tag:   tag,
	})

	for _, r := range results {
		if r.Success {
			sent++
		} else {
			failed++
		}
	}
	return sent, failed, nil
}
