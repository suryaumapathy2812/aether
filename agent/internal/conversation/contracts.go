package conversation

import "fmt"

type EventType string

const (
	EventStart               EventType = "start"
	EventStartStep           EventType = "start-step"
	EventTextDelta           EventType = "text-delta"
	EventToolInputAvailable  EventType = "tool-input-available"
	EventToolOutputAvailable EventType = "tool-output-available"
	EventFinishStep          EventType = "finish-step"
	EventFinish              EventType = "finish"
	EventError               EventType = "error"
)

type Event struct {
	SchemaVersion  string
	RequestID      string
	Sequence       int
	EventType      EventType
	IdempotencyKey string
	Payload        map[string]any
}

func NewEvent(requestID string, seq int, kind EventType, payload map[string]any) Event {
	if payload == nil {
		payload = map[string]any{}
	}
	return Event{
		SchemaVersion:  "1.0",
		RequestID:      requestID,
		Sequence:       seq,
		EventType:      kind,
		IdempotencyKey: fmt.Sprintf("%s:%d", requestID, seq),
		Payload:        payload,
	}
}
