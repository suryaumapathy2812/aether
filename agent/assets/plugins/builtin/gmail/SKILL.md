# Gmail Plugin

Read, send, draft, and manage emails on behalf of the user.

## Core Workflow

Gmail tools return **message IDs, not content**. The typical flow is always: find messages → read them → act on them.

```
Find IDs              →  Read content         →  Act
─────────────────────    ──────────────────      ──────────────────
inbox_count              read_gmail (parallel)   send_reply / reply
list_unread              get_thread              create_draft
search_email                                     archive_email
                                                 trash_email
                                                 add_label / remove_label
                                                 mark_read / mark_unread
```

### Why this matters

`list_unread` and `search_email` return `{messages: [{id, threadId}, ...], resultSizeEstimate, nextPageToken}`. The `resultSizeEstimate` is an approximation (can be off by 30-50%), and there's no subject, sender, or body. You need `read_gmail` for each ID to get actual email content. Never present raw message IDs to the user.

## Autonomy Rules

**Never ask the user what to search for.** You know what emails look like. When the user gives a broad request, figure out the search terms yourself:
- "Find my spending" → search for "transaction", "payment", "receipt", "order confirmation", "debit alert", common bank names (ICICI, HDFC, SBI, Axis), payment apps (Google Pay, PhonePe, Paytm, Amazon)
- "Emails from my boss" → check memory for boss's name/email, then search. If unknown, search recent emails and infer from frequency.
- "Important emails this week" → search for `is:important after:{date}` or `is:starred after:{date}`
- "Find that document someone sent me" → search for `has:attachment` with reasonable date range

**Always try at least 2-3 search strategies before saying you can't find something.** If the first search returns nothing useful, try different terms, broader date ranges, or different fields (subject vs sender vs body).

## Decision Rules

**Counting vs. listing:**
- "How many unread?" → `inbox_count` — it returns exact counts in one fast call
- "Show me my unread" → `list_unread` with `max_results=10`, then `read_gmail` in parallel for each ID
- "What's my latest email?" → `list_unread` with `max_results=1`, then `read_gmail`

**Searching vs. browsing:**
- General inbox check → `list_unread`
- Specific search (by sender, subject, date) → `search_email` with Gmail query syntax
- Broad discovery (spending, receipts, confirmations) → run multiple `search_email` calls with different queries in parallel, then merge and deduplicate results
- Query examples: `from:alice@example.com`, `subject:invoice`, `has:attachment`, `is:unread after:2024/01/01`, `label:Work`

**Before sending anything:**
- Confirm recipient, subject, and body with the user before calling `send_email` or `send_reply`. Sending is irreversible — drafts are not.
- Use `create_draft` freely without confirmation since drafts are safe and reversible.
- Use `send_reply` (not `send_email`) when replying to a thread — it keeps the conversation threaded in Gmail.
- Read the original email with `read_gmail` before composing a reply so you have context.

**Thread context:**
- Use `get_thread` to load the full conversation before replying to multi-message threads. This gives you the history needed to write a relevant reply.
- Thread IDs come from `list_unread` and `read_gmail` responses.

**Label management:**
- Call `list_labels` before `add_label` or `remove_label` to confirm the exact label name exists. Label names are case-insensitive in lookup, but use the exact name from `list_labels` when displaying to the user.

**Deletion:**
- Use `trash_email` for deletion — it's recoverable for 30 days. Let the user know they can recover it from Trash.

**Summarizing emails:**
- Keep summaries concise: "From Priya: asking about the Q3 report deadline"
- For long threads, summarize the most recent message and note the thread length.

## Parallel Reads

When reading multiple emails, call `read_gmail` in parallel (up to 25 concurrent). This is safe and dramatically faster than sequential reads. For bulk operations like marking 50 emails as read, batch in groups of 25.

## Pagination

`list_unread` and `search_email` responses include `nextPageToken` when more results exist. Pass it as `page_token` in follow-up calls. Set `max_results` based on what the user actually needs — don't over-fetch.

## Rate Limits

| Quota | Limit |
|---|---|
| Queries per minute per user | 250 |
| Send rate per user | 100 messages/day (2,000 for Workspace) |
| Concurrent requests | 25 per user |

## Example Workflows

**User: "Check my email"**
```
1. list_unread max_results=10 → get message IDs
2. read_gmail for each ID (parallel) → get subject, sender, body
3. Summarize: "You have 4 unread emails: [sender + subject for each]"
4. "Want me to read, reply, or archive any of them?"
```

**User: "Reply to Priya's email about the meeting"**
```
1. search_email query="from:priya meeting" max_results=5 → find the email
2. read_gmail message_id="..." → read full content for context
3. Draft a reply, show it to the user: "Here's what I'd send: [draft]. Send it?"
4. send_reply thread_id="..." to="priya@..." body="..."
```

**User: "Find all emails from my boss this month and label them as Work"**
```
1. search_email query="from:boss@company.com after:2026/03/01" max_results=20
2. read_gmail for each ID (parallel) → show summaries
3. list_labels → confirm "Work" label exists
4. add_label for each message_id with label_name="Work"
5. "Labeled 8 emails from your boss as 'Work'."
```
