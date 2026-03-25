---
name: gmail
description: Gmail API — send, read, search, label, archive emails
integration: google-workspace
---
# Gmail API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://gmail.googleapis.com/gmail/v1`
- Token auto-refreshes on 401 response

## List Messages
```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults=10&q=is:unread"
```
Response contains `messages[].id` — use each ID with Read Message below.

### Search Operators
- `is:unread` — unread messages
- `from:alice@example.com` — from a specific sender
- `subject:meeting` — subject contains word
- `newer_than:2d` — received in last 2 days
- `has:attachment` — messages with attachments

## Read Message
```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages/{MESSAGE_ID}?format=full"
```
- `format=full` returns headers + body (default)
- `format=metadata` returns headers only (faster)
- `format=raw` returns base64-encoded raw MIME

### Extracting the Body
The body is in `payload.parts[].body.data` (base64url-encoded). Decode with:
```bash
echo "ENCODED_DATA" | base64 -d 2>/dev/null || echo "ENCODED_DATA" | base64 -D
```
For multipart messages, check `payload.parts[].mimeType` for `text/plain` or `text/html`.

### Key Headers
- `From`, `To`, `Cc`, `Bcc`, `Subject`, `Date`
- Look in `payload.headers[]` array — each has `name` and `value`.

## Send Email
```bash
# 1. Build MIME message (base64url-encoded)
RAW=$(python3 -c "
import base64
msg = 'To: recipient@example.com\r\nSubject: Hello\r\n\r\nEmail body text here.'
print(base64.urlsafe_b64encode(msg.encode()).decode())
")

# 2. Send
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"raw\": \"$RAW\"}" \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
```

### Reply to a Message
```bash
RAW=$(python3 -c "
import base64
msg = '''To: original-sender@example.com
Subject: Re: Original Subject
In-Reply-To: <original-message-id>
References: <original-message-id>

Reply body text.'''
print(base64.urlsafe_b64encode(msg.encode()).decode())
")

curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"raw\": \"$RAW\", \"threadId\": \"THREAD_ID\"}" \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
```

## Modify Labels
```bash
# Add labels (e.g., mark as read by removing UNREAD)
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"removeLabelIds": ["UNREAD"]}' \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages/{MESSAGE_ID}/modify"

# Archive (remove INBOX label)
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"removeLabelIds": ["INBOX"]}' \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages/{MESSAGE_ID}/modify"
```

## List Labels
```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://gmail.googleapis.com/gmail/v1/users/me/labels"
```

## Trash / Untrash
```bash
# Trash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages/{MESSAGE_ID}/trash"

# Untrash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages/{MESSAGE_ID}/untrash"
```

## Rate Limits
- 250 quota units/user/second
- Sending: 100 messages/day (trial), 500/day (verified)
- Reading: effectively unlimited for normal usage

## Error Handling
- **401 Unauthorized**: Token expired — system auto-refreshes, retry the command
- **403 Forbidden**: Quota exceeded or insufficient permissions
- **404 Not Found**: Message ID doesn't exist or was deleted
- **429 Rate Limited**: Wait for `Retry-After` header value, then retry
