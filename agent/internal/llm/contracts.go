package llm

import (
	"fmt"
	"strconv"
	"time"
)

type EventType string

const (
	EventStart      EventType = "start"
	EventStartStep  EventType = "start-step"
	EventTextDelta  EventType = "text-delta"
	EventToolCall   EventType = "tool-call"
	EventToolResult EventType = "tool-result"
	EventFinishStep EventType = "finish-step"
	EventFinish     EventType = "finish"
	EventError      EventType = "error"
)

type LLMRequestEnvelope struct {
	SchemaVersion string
	RequestID     string
	JobID         string
	Kind          string
	Modality      string
	UserID        string
	SessionID     string
	Messages      []map[string]any
	Tools         []map[string]any
	ToolChoice    string
	PluginContext map[string]map[string]any
	Policy        map[string]any
	Trace         map[string]string
}

func (e LLMRequestEnvelope) Normalize() LLMRequestEnvelope {
	if e.SchemaVersion == "" {
		e.SchemaVersion = "1.0"
	}
	if e.RequestID == "" {
		e.RequestID = strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	if e.ToolChoice == "" {
		e.ToolChoice = "auto"
	}
	if e.Policy == nil {
		e.Policy = map[string]any{}
	}
	if e.PluginContext == nil {
		e.PluginContext = map[string]map[string]any{}
	}
	if e.Trace == nil {
		e.Trace = map[string]string{}
	}
	if e.Messages == nil {
		e.Messages = []map[string]any{}
	}
	if e.Tools == nil {
		e.Tools = []map[string]any{}
	}
	return e
}

type LLMEventEnvelope struct {
	SchemaVersion  string
	RequestID      string
	JobID          string
	EventType      EventType
	Sequence       int
	IdempotencyKey string
	Payload        map[string]any
	Metrics        map[string]any
}

func NewEvent(reqID, jobID string, kind EventType, seq int, payload map[string]any) LLMEventEnvelope {
	if payload == nil {
		payload = map[string]any{}
	}
	return LLMEventEnvelope{
		SchemaVersion:  "1.0",
		RequestID:      reqID,
		JobID:          jobID,
		EventType:      kind,
		Sequence:       seq,
		IdempotencyKey: fmt.Sprintf("%s:%d", reqID, seq),
		Payload:        payload,
		Metrics:        map[string]any{},
	}
}

type LLMResultEnvelope struct {
	SchemaVersion string
	RequestID     string
	JobID         string
	Status        string
	Output        map[string]any
	Usage         map[string]int
	Error         map[string]any
}
