package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

type ItemType string

const (
	ItemTypeMessage            ItemType = "message"
	ItemTypeFunctionCall       ItemType = "function_call"
	ItemTypeFunctionCallOutput ItemType = "function_call_output"
	ItemTypeReasoning          ItemType = "reasoning"
	ItemTypeText               ItemType = "text"
	ItemTypeAudio              ItemType = "audio"
)

type Item struct {
	Type     ItemType      `json:"type"`
	ID       string        `json:"id,omitempty"`
	Role     string        `json:"role,omitempty"`
	Content  string        `json:"content,omitempty"`
	Name     string        `json:"name,omitempty"`
	CallID   string        `json:"call_id,omitempty"`
	Function *Function     `json:"function,omitempty"`
	Input    any           `json:"input,omitempty"`
	Status   string        `json:"status,omitempty"`
	Parts    []ContentPart `json:"parts,omitempty"`
}

type ContentPart struct {
	Type     string      `json:"type"`
	Text     string      `json:"text,omitempty"`
	ImageURL *ImageURL   `json:"image_url,omitempty"`
	Audio    *AudioInput `json:"input_audio,omitempty"`
}

type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type AudioInput struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type Function struct {
	Name            string         `json:"name"`
	Arguments       string         `json:"arguments"`
	ArgumentsParsed map[string]any `json:"-"`
}

func (f *Function) ParseArguments() error {
	if f.Arguments == "" {
		f.ArgumentsParsed = map[string]any{}
		return nil
	}
	return json.Unmarshal([]byte(f.Arguments), &f.ArgumentsParsed)
}

type Reasoning struct {
	Summary  string `json:"summary"`
	Thinking string `json:"thinking,omitempty"`
}

func ParseItems(raw []map[string]any) ([]Item, error) {
	out := make([]Item, 0, len(raw))
	for i, msg := range raw {
		role, _ := msg["role"].(string)
		role = strings.TrimSpace(strings.ToLower(role))

		if toolCalls, ok := msg["tool_calls"].([]any); ok && len(toolCalls) > 0 {
			for _, tcRaw := range toolCalls {
				tc, ok := tcRaw.(map[string]any)
				if !ok {
					continue
				}
				fnRaw, _ := tc["function"].(map[string]any)
				fn := &Function{
					Name:      stringValue(fnRaw["name"]),
					Arguments: stringValue(fnRaw["arguments"]),
				}
				_ = fn.ParseArguments()
				item := Item{
					Type:     ItemTypeFunctionCall,
					ID:       stringValue(tc["id"]),
					Role:     "assistant",
					CallID:   stringValue(tc["id"]),
					Function: fn,
				}
				out = append(out, item)
			}
			if content, ok := msg["content"].(string); ok && strings.TrimSpace(content) != "" {
				out = append(out, Item{
					Type:    ItemTypeMessage,
					Role:    role,
					Content: content,
				})
			}
			continue
		}

		if role == "tool" {
			callID, _ := msg["tool_call_id"].(string)
			content, _ := msg["content"].(string)
			out = append(out, Item{
				Type:    ItemTypeFunctionCallOutput,
				Role:    "tool",
				CallID:  callID,
				Content: content,
				Status:  "completed",
			})
			continue
		}

		if role == "system" || role == "user" || role == "assistant" || role == "" {
			item := Item{
				Type: ItemTypeMessage,
				Role: role,
			}
			if content, ok := msg["content"].(string); ok {
				item.Content = content
			} else if parts, ok := msg["content"].([]any); ok {
				item.Parts = parseContentParts(parts)
			}
			out = append(out, item)
			continue
		}

		if rawContent, ok := msg["content"].(string); ok && strings.TrimSpace(rawContent) != "" {
			out = append(out, Item{
				Type:    ItemTypeMessage,
				Role:    role,
				Content: rawContent,
			})
			continue
		}

		return nil, fmt.Errorf("items[%d]: unsupported message format", i)
	}
	return out, nil
}

func parseContentParts(parts []any) []ContentPart {
	out := make([]ContentPart, 0, len(parts))
	for _, p := range parts {
		pmap, ok := p.(map[string]any)
		if !ok {
			continue
		}
		typ, _ := pmap["type"].(string)
		part := ContentPart{Type: typ}
		switch typ {
		case "text":
			part.Text, _ = pmap["text"].(string)
		case "image_url":
			if iu, ok := pmap["image_url"].(map[string]any); ok {
				part.ImageURL = &ImageURL{
					URL:    stringValue(iu["url"]),
					Detail: stringValue(iu["detail"]),
				}
			}
		case "input_audio":
			if aa, ok := pmap["input_audio"].(map[string]any); ok {
				part.Audio = &AudioInput{
					Data:   stringValue(aa["data"]),
					Format: stringValue(aa["format"]),
				}
			}
		}
		out = append(out, part)
	}
	return out
}

func (i *Item) ToChatMessage() map[string]any {
	switch i.Type {
	case ItemTypeFunctionCall:
		msg := map[string]any{"role": "assistant"}
		if i.Function != nil {
			args, _ := json.Marshal(i.Function.ArgumentsParsed)
			msg["tool_calls"] = []map[string]any{{
				"id":   i.CallID,
				"type": "function",
				"function": map[string]any{
					"name":      i.Function.Name,
					"arguments": string(args),
				},
			}}
		}
		if i.Content != "" {
			msg["content"] = i.Content
		}
		return msg
	case ItemTypeFunctionCallOutput:
		return map[string]any{
			"role":         "tool",
			"tool_call_id": i.CallID,
			"content":      i.Content,
		}
	case ItemTypeMessage:
		msg := map[string]any{"role": i.Role}
		if len(i.Parts) > 0 {
			msg["content"] = partsToContent(i.Parts)
		} else {
			msg["content"] = i.Content
		}
		return msg
	default:
		return map[string]any{
			"role":    i.Role,
			"content": i.Content,
		}
	}
}

func partsToContent(parts []ContentPart) []any {
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case "text":
			out = append(out, map[string]any{"type": "text", "text": p.Text})
		case "image_url":
			if p.ImageURL != nil {
				out = append(out, map[string]any{"type": "image_url", "image_url": p.ImageURL})
			}
		case "input_audio":
			if p.Audio != nil {
				out = append(out, map[string]any{"type": "input_audio", "input_audio": p.Audio})
			}
		}
	}
	return out
}

func (i *Item) ToResponsesInput() map[string]any {
	switch i.Type {
	case ItemTypeFunctionCall:
		out := map[string]any{
			"type": "function_call",
			"name": i.Function.Name,
			"id":   i.CallID,
		}
		if i.Function.Arguments != "" {
			out["arguments"] = i.Function.Arguments
		}
		return out
	case ItemTypeFunctionCallOutput:
		return map[string]any{
			"type":    "function_call_output",
			"call_id": i.CallID,
			"output":  i.Content,
		}
	case ItemTypeMessage:
		if i.Role == "system" {
			return map[string]any{
				"type":    "message",
				"role":    "system",
				"content": i.Content,
			}
		}
		if len(i.Parts) > 0 {
			return map[string]any{
				"type":    "message",
				"role":    i.Role,
				"content": partsToContent(i.Parts),
			}
		}
		return map[string]any{
			"type":    "message",
			"role":    i.Role,
			"content": i.Content,
		}
	default:
		return map[string]any{
			"type":    "message",
			"role":    i.Role,
			"content": i.Content,
		}
	}
}

func ItemsToChatMessages(items []Item) []map[string]any {
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		out = append(out, item.ToChatMessage())
	}
	return out
}

func ItemsToResponsesInput(items []Item) []map[string]any {
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		out = append(out, item.ToResponsesInput())
	}
	return out
}

func ChatMessagesToItems(messages []map[string]any) ([]Item, error) {
	return ParseItems(messages)
}

func stringValue(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}
