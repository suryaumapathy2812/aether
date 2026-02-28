package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

const gmailAPI = "https://gmail.googleapis.com/gmail/v1/users/me"

type ListUnreadTool struct{}
type ReadEmailTool struct{}
type SendReplyTool struct{}
type SendEmailTool struct{}
type CreateDraftTool struct{}
type ArchiveEmailTool struct{}
type SearchEmailTool struct{}
type ReplyAllTool struct{}
type GetThreadTool struct{}
type TrashEmailTool struct{}
type MarkReadTool struct{}
type MarkUnreadTool struct{}
type ListLabelsTool struct{}
type AddLabelTool struct{}
type RemoveLabelTool struct{}
type RefreshGmailTokenTool struct{}

func (t *ListUnreadTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_unread", Description: "List unread Gmail messages.", StatusText: "Loading unread emails...", Parameters: []tools.Param{{Name: "max_results", Type: "integer", Required: false, Default: 10}}}
}

func (t *ListUnreadTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	max, _ := asInt(call.Args["max_results"])
	if max <= 0 {
		max = 10
	}
	v := url.Values{}
	v.Set("q", "is:unread")
	v.Set("maxResults", fmt.Sprintf("%d", max))
	obj, err := gmailRequest(ctx, call, http.MethodGet, gmailAPI+"/messages?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["messages"])
	return tools.Success(string(b), map[string]any{"max_results": max})
}

func (t *ReadEmailTool) Definition() tools.Definition {
	return tools.Definition{Name: "read_gmail", Description: "Read a Gmail message by message id.", StatusText: "Reading email...", Parameters: []tools.Param{{Name: "message_id", Type: "string", Required: true}}}
}

func (t *ReadEmailTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	msgID, _ := call.Args["message_id"].(string)
	obj, err := gmailRequest(ctx, call, http.MethodGet, gmailAPI+"/messages/"+url.PathEscape(msgID)+"?format=full", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj)
	return tools.Success(string(b), map[string]any{"message_id": msgID})
}

func (t *SendReplyTool) Definition() tools.Definition {
	return tools.Definition{Name: "send_reply", Description: "Send a Gmail reply in an existing thread.", StatusText: "Sending reply...", Parameters: []tools.Param{{Name: "thread_id", Type: "string", Required: true}, {Name: "to", Type: "string", Required: true}, {Name: "body", Type: "string", Required: true}, {Name: "subject", Type: "string", Required: false}}}
}

func (t *SendReplyTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	threadID, _ := call.Args["thread_id"].(string)
	to, _ := call.Args["to"].(string)
	body, _ := call.Args["body"].(string)
	subject, _ := call.Args["subject"].(string)
	if strings.TrimSpace(subject) == "" {
		subject = "Re: (no subject)"
	}
	raw := base64URL("To: " + to + "\r\nSubject: " + subject + "\r\n\r\n" + body)
	obj, err := gmailRequest(ctx, call, http.MethodPost, gmailAPI+"/messages/send", map[string]any{"raw": raw, "threadId": threadID})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Reply sent.", map[string]any{"thread_id": threadID, "message_id": obj["id"]})
}

func (t *SendEmailTool) Definition() tools.Definition {
	return tools.Definition{Name: "send_email", Description: "Send a Gmail email.", StatusText: "Sending email...", Parameters: []tools.Param{{Name: "to", Type: "string", Required: true}, {Name: "subject", Type: "string", Required: true}, {Name: "body", Type: "string", Required: true}, {Name: "cc", Type: "string", Required: false}, {Name: "bcc", Type: "string", Required: false}}}
}

func (t *SendEmailTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	to, _ := call.Args["to"].(string)
	subject, _ := call.Args["subject"].(string)
	body, _ := call.Args["body"].(string)
	cc, _ := call.Args["cc"].(string)
	bcc, _ := call.Args["bcc"].(string)
	headers := []string{"To: " + to, "Subject: " + subject}
	if strings.TrimSpace(cc) != "" {
		headers = append(headers, "Cc: "+cc)
	}
	if strings.TrimSpace(bcc) != "" {
		headers = append(headers, "Bcc: "+bcc)
	}
	raw := base64URL(strings.Join(headers, "\r\n") + "\r\n\r\n" + body)
	obj, err := gmailRequest(ctx, call, http.MethodPost, gmailAPI+"/messages/send", map[string]any{"raw": raw})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Email sent.", map[string]any{"message_id": obj["id"]})
}

func (t *CreateDraftTool) Definition() tools.Definition {
	return tools.Definition{Name: "create_draft", Description: "Create a Gmail draft.", StatusText: "Creating draft...", Parameters: []tools.Param{{Name: "to", Type: "string", Required: true}, {Name: "subject", Type: "string", Required: true}, {Name: "body", Type: "string", Required: true}, {Name: "cc", Type: "string", Required: false}, {Name: "bcc", Type: "string", Required: false}}}
}

func (t *CreateDraftTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	to, _ := call.Args["to"].(string)
	subject, _ := call.Args["subject"].(string)
	body, _ := call.Args["body"].(string)
	cc, _ := call.Args["cc"].(string)
	bcc, _ := call.Args["bcc"].(string)
	headers := []string{"To: " + to, "Subject: " + subject}
	if cc != "" {
		headers = append(headers, "Cc: "+cc)
	}
	if bcc != "" {
		headers = append(headers, "Bcc: "+bcc)
	}
	raw := base64URL(strings.Join(headers, "\r\n") + "\r\n\r\n" + body)
	obj, err := gmailRequest(ctx, call, http.MethodPost, gmailAPI+"/drafts", map[string]any{"message": map[string]any{"raw": raw}})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Draft created.", map[string]any{"draft_id": obj["id"]})
}

func (t *ArchiveEmailTool) Definition() tools.Definition {
	return tools.Definition{Name: "archive_email", Description: "Archive a Gmail message.", StatusText: "Archiving email...", Parameters: []tools.Param{{Name: "message_id", Type: "string", Required: true}}}
}

func (t *ArchiveEmailTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return gmailModifyLabels(ctx, call, "removeLabelIds", []string{"INBOX"}, "Email archived.")
}

func (t *SearchEmailTool) Definition() tools.Definition {
	return tools.Definition{Name: "search_email", Description: "Search Gmail by query.", StatusText: "Searching email...", Parameters: []tools.Param{{Name: "query", Type: "string", Required: true}, {Name: "max_results", Type: "integer", Required: false, Default: 10}}}
}

func (t *SearchEmailTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	query, _ := call.Args["query"].(string)
	max, _ := asInt(call.Args["max_results"])
	if max <= 0 {
		max = 10
	}
	v := url.Values{}
	v.Set("q", query)
	v.Set("maxResults", fmt.Sprintf("%d", max))
	obj, err := gmailRequest(ctx, call, http.MethodGet, gmailAPI+"/messages?"+v.Encode(), nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["messages"])
	return tools.Success(string(b), map[string]any{"query": query})
}

func (t *ReplyAllTool) Definition() tools.Definition {
	return tools.Definition{Name: "reply_all", Description: "Reply to all recipients for an email.", StatusText: "Sending reply-all...", Parameters: []tools.Param{{Name: "message_id", Type: "string", Required: true}, {Name: "body", Type: "string", Required: true}, {Name: "subject", Type: "string", Required: false}}}
}

func (t *ReplyAllTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	msgID, _ := call.Args["message_id"].(string)
	body, _ := call.Args["body"].(string)
	subject, _ := call.Args["subject"].(string)
	meta, err := gmailRequest(ctx, call, http.MethodGet, gmailAPI+"/messages/"+url.PathEscape(msgID)+"?format=metadata&metadataHeaders=From&metadataHeaders=To&metadataHeaders=Cc&metadataHeaders=Subject", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	payload, _ := meta["payload"].(map[string]any)
	headers, _ := payload["headers"].([]any)
	from, to, cc, subj := "", "", "", subject
	for _, h := range headers {
		m, _ := h.(map[string]any)
		name, _ := m["name"].(string)
		val, _ := m["value"].(string)
		switch strings.ToLower(name) {
		case "from":
			from = val
		case "to":
			to = val
		case "cc":
			cc = val
		case "subject":
			if strings.TrimSpace(subj) == "" {
				subj = val
			}
		}
	}
	if !strings.HasPrefix(strings.ToLower(subj), "re:") {
		subj = "Re: " + subj
	}
	raw := base64URL("To: " + from + "\r\nCc: " + strings.Trim(strings.Join([]string{to, cc}, ", "), ", ") + "\r\nSubject: " + subj + "\r\n\r\n" + body)
	threadID, _ := meta["threadId"].(string)
	obj, err := gmailRequest(ctx, call, http.MethodPost, gmailAPI+"/messages/send", map[string]any{"raw": raw, "threadId": threadID})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Reply-all sent.", map[string]any{"message_id": obj["id"], "thread_id": threadID})
}

func (t *GetThreadTool) Definition() tools.Definition {
	return tools.Definition{Name: "get_thread", Description: "Get all messages in a Gmail thread.", StatusText: "Loading thread...", Parameters: []tools.Param{{Name: "thread_id", Type: "string", Required: true}}}
}

func (t *GetThreadTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	threadID, _ := call.Args["thread_id"].(string)
	obj, err := gmailRequest(ctx, call, http.MethodGet, gmailAPI+"/threads/"+url.PathEscape(threadID)+"?format=full", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["messages"])
	return tools.Success(string(b), map[string]any{"thread_id": threadID})
}

func (t *TrashEmailTool) Definition() tools.Definition {
	return tools.Definition{Name: "trash_email", Description: "Move email to Gmail trash.", StatusText: "Moving email to trash...", Parameters: []tools.Param{{Name: "message_id", Type: "string", Required: true}}}
}

func (t *TrashEmailTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	msgID, _ := call.Args["message_id"].(string)
	_, err := gmailRequest(ctx, call, http.MethodPost, gmailAPI+"/messages/"+url.PathEscape(msgID)+"/trash", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Email moved to trash.", map[string]any{"message_id": msgID})
}

func (t *MarkReadTool) Definition() tools.Definition {
	return tools.Definition{Name: "mark_read", Description: "Mark Gmail message as read.", StatusText: "Marking as read...", Parameters: []tools.Param{{Name: "message_id", Type: "string", Required: true}}}
}

func (t *MarkReadTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return gmailModifyLabels(ctx, call, "removeLabelIds", []string{"UNREAD"}, "Email marked as read.")
}

func (t *MarkUnreadTool) Definition() tools.Definition {
	return tools.Definition{Name: "mark_unread", Description: "Mark Gmail message as unread.", StatusText: "Marking as unread...", Parameters: []tools.Param{{Name: "message_id", Type: "string", Required: true}}}
}

func (t *MarkUnreadTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return gmailModifyLabels(ctx, call, "addLabelIds", []string{"UNREAD"}, "Email marked as unread.")
}

func (t *ListLabelsTool) Definition() tools.Definition {
	return tools.Definition{Name: "list_labels", Description: "List Gmail labels.", StatusText: "Fetching labels..."}
}

func (t *ListLabelsTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	obj, err := gmailRequest(ctx, call, http.MethodGet, gmailAPI+"/labels", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	b, _ := json.Marshal(obj["labels"])
	return tools.Success(string(b), nil)
}

func (t *AddLabelTool) Definition() tools.Definition {
	return tools.Definition{Name: "add_label", Description: "Add Gmail label to message.", StatusText: "Adding label...", Parameters: []tools.Param{{Name: "message_id", Type: "string", Required: true}, {Name: "label_name", Type: "string", Required: true}}}
}

func (t *AddLabelTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return gmailModifyByLabelName(ctx, call, "addLabelIds")
}

func (t *RemoveLabelTool) Definition() tools.Definition {
	return tools.Definition{Name: "remove_label", Description: "Remove Gmail label from message.", StatusText: "Removing label...", Parameters: []tools.Param{{Name: "message_id", Type: "string", Required: true}, {Name: "label_name", Type: "string", Required: true}}}
}

func (t *RemoveLabelTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return gmailModifyByLabelName(ctx, call, "removeLabelIds")
}

func (t *RefreshGmailTokenTool) Definition() tools.Definition {
	return tools.Definition{Name: "refresh_gmail_token", Description: "Refresh Gmail OAuth access token.", StatusText: "Refreshing Gmail token..."}
}

func (t *RefreshGmailTokenTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	return refreshOAuthAccessToken(ctx, call, "https://oauth2.googleapis.com/token", false)
}

func gmailModifyLabels(ctx context.Context, call tools.Call, key string, values []string, okMsg string) tools.Result {
	msgID, _ := call.Args["message_id"].(string)
	_, err := gmailRequest(ctx, call, http.MethodPost, gmailAPI+"/messages/"+url.PathEscape(msgID)+"/modify", map[string]any{key: values})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success(okMsg, map[string]any{"message_id": msgID})
}

func gmailModifyByLabelName(ctx context.Context, call tools.Call, key string) tools.Result {
	msgID, _ := call.Args["message_id"].(string)
	labelName, _ := call.Args["label_name"].(string)
	labelsObj, err := gmailRequest(ctx, call, http.MethodGet, gmailAPI+"/labels", nil)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	labelID := ""
	if labels, ok := labelsObj["labels"].([]any); ok {
		for _, raw := range labels {
			m, _ := raw.(map[string]any)
			name, _ := m["name"].(string)
			if strings.EqualFold(name, labelName) {
				labelID, _ = m["id"].(string)
				break
			}
		}
	}
	if labelID == "" {
		return tools.Fail("Label not found: "+labelName, nil)
	}
	_, err = gmailRequest(ctx, call, http.MethodPost, gmailAPI+"/messages/"+url.PathEscape(msgID)+"/modify", map[string]any{key: []string{labelID}})
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	return tools.Success("Label updated.", map[string]any{"message_id": msgID, "label_name": labelName})
}

func gmailRequest(ctx context.Context, call tools.Call, method, reqURL string, body any) (map[string]any, error) {
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return nil, err
	}
	token, err := requireToken(ctx, call, cfg)
	if err != nil {
		return nil, fmt.Errorf("Gmail is not connected: %w", err)
	}
	var reqBody *strings.Reader
	if body == nil {
		reqBody = strings.NewReader("")
	} else {
		b, _ := json.Marshal(body)
		reqBody = strings.NewReader(string(b))
	}
	req, err := http.NewRequestWithContext(ctx, method, reqURL, reqBody)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := (&http.Client{Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Gmail API request failed with status %d", resp.StatusCode)
	}
	if resp.StatusCode == http.StatusNoContent {
		return map[string]any{}, nil
	}
	var obj map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&obj); err != nil {
		return nil, err
	}
	return obj, nil
}

var (
	_ tools.Tool = (*ListUnreadTool)(nil)
	_ tools.Tool = (*ReadEmailTool)(nil)
	_ tools.Tool = (*SendReplyTool)(nil)
	_ tools.Tool = (*SendEmailTool)(nil)
	_ tools.Tool = (*CreateDraftTool)(nil)
	_ tools.Tool = (*ArchiveEmailTool)(nil)
	_ tools.Tool = (*SearchEmailTool)(nil)
	_ tools.Tool = (*ReplyAllTool)(nil)
	_ tools.Tool = (*GetThreadTool)(nil)
	_ tools.Tool = (*TrashEmailTool)(nil)
	_ tools.Tool = (*MarkReadTool)(nil)
	_ tools.Tool = (*MarkUnreadTool)(nil)
	_ tools.Tool = (*ListLabelsTool)(nil)
	_ tools.Tool = (*AddLabelTool)(nil)
	_ tools.Tool = (*RemoveLabelTool)(nil)
	_ tools.Tool = (*RefreshGmailTokenTool)(nil)
)
