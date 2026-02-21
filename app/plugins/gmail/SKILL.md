# Gmail Plugin

You have access to Gmail tools for reading, sending, drafting, and managing emails on behalf of the user.

---

## Tools Available

### `list_unread`
Fetch unread emails from the inbox. Returns subject, sender, date, snippet, message ID, and thread ID for each message.

**Parameters:** None

**Use when:** The user asks about new emails, wants to check their inbox, or you need message IDs before reading or replying.

---

### `read_gmail`
Read the full content of a specific email by its message ID.

**Parameters:**
- `message_id` (required) — The message ID from `list_unread` or a prior search

**Returns:** Full headers (From, To, Subject, Date) and decoded body text.

**Use when:** You need the full content of an email before replying, summarizing, or acting on it.

---

### `send_email`
Send a new email immediately.

**Parameters:**
- `to` (required) — Recipient email address(es), comma-separated
- `subject` (required) — Email subject line
- `body` (required) — Email body text (plain text)
- `cc` (optional) — CC recipients, comma-separated
- `bcc` (optional) — BCC recipients, comma-separated

**Use when:** The user explicitly asks to send an email. **Always confirm recipient, subject, and body before calling this tool.**

---

### `send_reply`
Reply to an existing email thread.

**Parameters:**
- `thread_id` (required) — Thread ID from the original email (keeps the conversation threaded)
- `to` (required) — Recipient email address
- `body` (required) — Reply body text
- `subject` (optional) — Subject line (defaults to "Re: [original subject]")

**Use when:** The user wants to reply to a specific email. Use the thread ID from `list_unread` or `read_gmail` to keep it threaded.

---

### `create_draft`
Save an email as a draft in Gmail Drafts (not sent).

**Parameters:**
- `to` (required) — Recipient email address(es)
- `subject` (required) — Email subject line
- `body` (required) — Email body text
- `cc` (optional) — CC recipients
- `bcc` (optional) — BCC recipients

**Use when:** The user wants to draft an email for review before sending, or says "draft" instead of "send".

---

### `archive_email`
Archive an email (removes it from inbox, keeps it in All Mail).

**Parameters:**
- `message_id` (required) — The message ID to archive

**Use when:** The user wants to clean up their inbox, or you've handled a message and the user wants it out of the way.

---

### `search_email`
Search Gmail using Gmail query syntax.

**Parameters:**
- `query` (required) — Gmail search query (e.g. `from:boss@company.com`, `subject:invoice`, `has:attachment`, `is:unread after:2024/01/01`)
- `max_results` (optional, default 10) — Max emails to return

**Returns:** Numbered list with subject, sender, date, and message ID for each result.

**Use when:** The user wants to find specific emails by sender, subject, keyword, date, or any Gmail filter criteria. Prefer this over `list_unread` when the user has a specific search in mind.

**Query syntax examples:**
- `from:alice@example.com` — emails from a specific sender
- `subject:invoice` — emails with "invoice" in the subject
- `has:attachment` — emails with attachments
- `is:unread` — unread emails only
- `after:2024/01/01 before:2024/02/01` — date range
- `label:Work` — emails with a specific label

---

### `reply_all`
Reply to all recipients of an email (sender + all To/CC recipients).

**Parameters:**
- `message_id` (required) — The message ID to reply to
- `body` (required) — Reply body text
- `subject` (optional) — Subject override (defaults to Re: [original subject])

**Use when:** The user explicitly says "reply all", or the email has multiple recipients and the context clearly indicates all should be included in the reply. **Always confirm the full recipient list with the user before sending.**

---

### `get_thread`
Get all messages in an email thread.

**Parameters:**
- `thread_id` (required) — Gmail thread ID

**Returns:** Each message with sender, date, message ID, and a 200-character preview snippet.

**Use when:** The user wants to read a full conversation, or you need full context before composing a reply. Use this before `reply_all` or `send_reply` when the thread has multiple messages.

---

### `trash_email`
Move an email to trash (recoverable within 30 days).

**Parameters:**
- `message_id` (required) — The message ID to trash

**Use when:** The user wants to delete an email. **Always use `trash_email` instead of any permanent delete** — it's reversible. Inform the user the email can be recovered from Trash within 30 days.

---

### `mark_read`
Mark an email as read.

**Parameters:**
- `message_id` (required) — The message ID to mark as read

**Use when:** The user wants to mark an email as read, or after reading/processing an email the user wants cleared from their unread count.

---

### `mark_unread`
Mark an email as unread.

**Parameters:**
- `message_id` (required) — The message ID to mark as unread

**Use when:** The user wants to flag an email to come back to it later, or explicitly asks to mark something as unread.

---

### `list_labels`
List all Gmail labels (folders).

**Parameters:** None

**Returns:** All user-defined labels with their names and IDs (system CATEGORY_ labels are filtered out).

**Use when:** The user asks what labels/folders they have, or before calling `add_label` or `remove_label` to confirm the exact label name.

---

### `add_label`
Add a label to an email.

**Parameters:**
- `message_id` (required) — The message ID
- `label_name` (required) — The label name (e.g. `Work`, `Important`) — not the ID

**Use when:** The user wants to categorize or tag an email. **Always call `list_labels` first** to confirm the exact label name before adding it.

---

### `remove_label`
Remove a label from an email.

**Parameters:**
- `message_id` (required) — The message ID
- `label_name` (required) — The label name to remove

**Use when:** The user wants to remove a category/tag from an email. **Always call `list_labels` first** to confirm the exact label name.

---

## Decision Rules

**Reading emails:**
- Always call `list_unread` first to get message IDs before reading specific emails
- Call `read_gmail` when you need the full body — the snippet from `list_unread` is often enough for summaries
- Reference the sender by name if available from the headers

**Sending emails:**
- **Always confirm with the user before calling `send_email` or `send_reply`** — show them the recipient, subject, and body first
- Use `create_draft` without confirmation — drafts are safe and reversible
- Use `send_reply` (not `send_email`) when replying to an existing thread — it keeps the conversation threaded in Gmail
- For replies, always read the original email first with `read_gmail` to understand the context

**Summarizing emails:**
- Keep summaries concise — one or two sentences per email
- Lead with who it's from and what it's about: "From Priya: she's asking about the Q3 report deadline"
- For long threads, summarize the most recent message and note how many messages are in the thread

**Inbox management:**
- Suggest archiving promotional or handled emails to keep the inbox clean
- Don't archive without asking unless the user has given a standing instruction to do so
- **Always use `trash_email` for deletion** — it's reversible. Never permanently delete.

**Searching emails:**
- Use `search_email` with Gmail query syntax for targeted searches (by sender, subject, date, label, etc.)
- Use `list_unread` for a general "check my inbox" request
- Combine search operators for precision: `from:alice@example.com subject:report is:unread`

**Reply-all:**
- Use `reply_all` only when the user explicitly says "reply all" or when the email clearly has multiple recipients who all need to be included
- **Always show the full recipient list to the user before calling `reply_all`** — it's easy to accidentally include unintended recipients
- Use `get_thread` to read the full conversation before composing a reply-all

**Thread management:**
- Use `get_thread` to load full conversation context before replying to a multi-message thread
- The thread ID is available from `list_unread` and `read_gmail` responses

**Label management:**
- **Always call `list_labels` first** to confirm the exact label name before calling `add_label` or `remove_label`
- Label names are case-insensitive in the lookup, but show the user the exact name from `list_labels`
- If a label doesn't exist, suggest the user create it in Gmail settings

**Error handling:**
- If `list_unread` returns empty, tell the user their inbox is clear — don't assume an error
- If a message ID is invalid, ask the user to clarify which email they mean
- If `search_email` returns no results, suggest refining the query or checking the date range

---

## Example Workflows

**"Check my email"**
```
1. list_unread → get list of unread messages
2. Summarize each: sender, subject, brief snippet
3. Ask if they want to read, reply, or archive any of them
```

**"Reply to Priya's email about the meeting"**
```
1. list_unread → find Priya's email, get message_id and thread_id
2. read_gmail message_id="..." → read full content for context
3. Draft a reply, show it to the user for confirmation
4. send_reply thread_id="..." to="priya@..." body="..."
```

**"Draft an email to the team about tomorrow's standup"**
```
1. create_draft to="team@..." subject="Standup tomorrow" body="..."
2. Confirm: "Draft saved. Want me to send it or make any changes?"
```

**"Find all emails from my boss this month"**
```
1. search_email query="from:boss@company.com after:2026/02/01"
2. List results with subject, date, and message ID
3. Ask if they want to read any of them
```

**"Reply all to the project thread"**
```
1. search_email or list_unread → find the email, get message_id
2. get_thread thread_id="..." → read full conversation for context
3. Draft the reply-all, show the user the full recipient list for confirmation
4. reply_all message_id="..." body="..."
```

**"Delete this email"**
```
1. trash_email message_id="..."
2. "Moved to Trash. You can recover it within 30 days if needed."
```

**"Label this email as Work"**
```
1. list_labels → confirm "Work" label exists and get exact name
2. add_label message_id="..." label_name="Work"
3. Confirm: "Added the 'Work' label to that email."
```

**"Mark all these emails as read"**
```
1. For each message_id: mark_read message_id="..."
2. Confirm: "Marked X emails as read."
```
