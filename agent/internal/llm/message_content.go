package llm

import "strings"

// ExtractTextFromMessageContent returns user-visible text from either a plain
// string content field or multimodal content parts.
func ExtractTextFromMessageContent(content any) string {
	if text, ok := content.(string); ok {
		return strings.TrimSpace(text)
	}
	parts, ok := content.([]any)
	if !ok {
		return ""
	}
	chunks := make([]string, 0, len(parts))
	for _, part := range parts {
		m, ok := part.(map[string]any)
		if !ok {
			continue
		}
		typ, _ := m["type"].(string)
		if typ != "text" {
			continue
		}
		v, _ := m["text"].(string)
		v = strings.TrimSpace(v)
		if v != "" {
			chunks = append(chunks, v)
		}
	}
	return strings.TrimSpace(strings.Join(chunks, "\n"))
}

func LatestUserMessageText(messages []map[string]any) string {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		role, _ := msg["role"].(string)
		if role != "user" {
			continue
		}
		if text := ExtractTextFromMessageContent(msg["content"]); text != "" {
			return text
		}
	}
	return ""
}

func LatestUserMessageContent(messages []map[string]any) any {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		role, _ := msg["role"].(string)
		if role != "user" {
			continue
		}
		if content, ok := msg["content"]; ok {
			return content
		}
		return nil
	}
	return nil
}

// LatestUserTurnSummary returns a summary for the most recent user turn only.
// For image/audio-only turns, it returns a compact marker so conversation
// records still reflect the actual last user message.
func LatestUserTurnSummary(messages []map[string]any) string {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		role, _ := msg["role"].(string)
		if role != "user" {
			continue
		}
		content := msg["content"]
		if text := ExtractTextFromMessageContent(content); text != "" {
			return text
		}
		hasImage, hasAudio := ContentHasMediaKinds(content)
		switch {
		case hasImage && hasAudio:
			return "[image+audio]"
		case hasImage:
			return "[image]"
		case hasAudio:
			return "[audio]"
		default:
			return ""
		}
	}
	return ""
}

func ContentHasMediaKinds(content any) (hasImage bool, hasAudio bool) {
	parts, ok := content.([]any)
	if !ok {
		return false, false
	}
	for _, part := range parts {
		m, ok := part.(map[string]any)
		if !ok {
			continue
		}
		typ, _ := m["type"].(string)
		switch typ {
		case "image_url":
			hasImage = true
		case "input_audio":
			hasAudio = true
		}
	}
	return hasImage, hasAudio
}
