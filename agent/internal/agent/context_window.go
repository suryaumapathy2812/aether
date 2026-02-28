package agent

import (
	"encoding/json"
	"strings"
)

type ContextWindowManager struct {
	SoftChars int
	HardChars int
	KeepTail  int
}

func DefaultContextWindow() ContextWindowManager {
	return ContextWindowManager{SoftChars: 42000, HardChars: 52000, KeepTail: 24}
}

func (m ContextWindowManager) Apply(messages []map[string]any) ([]map[string]any, string) {
	if len(messages) == 0 {
		return messages, ""
	}
	soft := m.SoftChars
	hard := m.HardChars
	if soft <= 0 {
		soft = 42000
	}
	if hard <= 0 {
		hard = 52000
	}
	if estimateChars(messages) <= soft {
		return messages, ""
	}

	trimmed := cloneMessages(messages)
	for i := 0; i < len(trimmed); i++ {
		role, _ := trimmed[i]["role"].(string)
		if role == "tool" {
			content, _ := trimmed[i]["content"].(string)
			if len(content) > 1200 {
				trimmed[i]["content"] = content[:1200] + "\n...truncated for context budget"
			}
		}
	}
	if estimateChars(trimmed) <= hard {
		return trimmed, "Truncated large tool outputs to fit context budget."
	}

	keepTail := m.KeepTail
	if keepTail <= 0 {
		keepTail = 24
	}
	if keepTail >= len(trimmed) {
		keepTail = len(trimmed) / 2
	}
	if keepTail <= 0 {
		keepTail = 1
	}

	head := []map[string]any{}
	tailStart := len(trimmed) - keepTail
	for i, msg := range trimmed {
		if i == 0 {
			if role, _ := msg["role"].(string); role == "system" {
				head = append(head, msg)
			}
		}
		if i >= tailStart {
			break
		}
	}
	middle := trimmed[len(head):tailStart]
	summary := summarizeMiddle(middle)
	result := make([]map[string]any, 0, len(head)+keepTail+1)
	result = append(result, head...)
	result = append(result, map[string]any{"role": "system", "content": summary})
	result = append(result, trimmed[tailStart:]...)
	return result, "Compacted older messages into a summary to fit context window."
}

func summarizeMiddle(messages []map[string]any) string {
	if len(messages) == 0 {
		return "Conversation summary: earlier context compacted."
	}
	lines := []string{"Conversation summary of earlier turns:"}
	for _, msg := range messages {
		role, _ := msg["role"].(string)
		content := messageContent(msg)
		if content == "" {
			continue
		}
		if len(content) > 220 {
			content = content[:220] + "..."
		}
		lines = append(lines, "- "+role+": "+content)
		if len(strings.Join(lines, "\n")) > 2000 {
			break
		}
	}
	return strings.Join(lines, "\n")
}

func messageContent(m map[string]any) string {
	if s, ok := m["content"].(string); ok {
		return strings.TrimSpace(s)
	}
	b, _ := json.Marshal(m["content"])
	return strings.TrimSpace(string(b))
}

func estimateChars(messages []map[string]any) int {
	n := 0
	for _, msg := range messages {
		b, _ := json.Marshal(msg)
		n += len(b)
	}
	return n
}

func cloneMessages(messages []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		cpy := map[string]any{}
		for k, v := range msg {
			cpy[k] = v
		}
		out = append(out, cpy)
	}
	return out
}
