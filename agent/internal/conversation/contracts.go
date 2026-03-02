package conversation

import "fmt"

type EventType string

const (
	EventAck        EventType = "ack"
	EventAnswer     EventType = "answer"
	EventToolCall   EventType = "tool_call"
	EventToolResult EventType = "tool_result"
	EventStatus     EventType = "status"
	EventError      EventType = "error"
	EventDone       EventType = "done"
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
