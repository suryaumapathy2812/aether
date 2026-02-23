# Telegram Plugin

You can send and receive Telegram messages on behalf of the user via their Telegram bot.

---

## Tools Available

### `handle_telegram_event`
Process an incoming Telegram webhook update. Extracts the sender, chat ID, message text, and media type.

**Parameters:**
- `payload` (required) — Raw Telegram Update object (passed automatically from the webhook instruction)

**Returns:** A structured summary with `chat_id`, `sender_name`, `text`, `message_id`, and `chat_type`.

**Use when:** You receive an instruction saying a Telegram message has arrived. Always call this tool first to extract the message details before deciding how to respond.

---

### `telegram_send_message`
Send a text message to a Telegram chat.

**Parameters:**
- `chat_id` (required) — The Telegram chat ID to send to (use the `chat_id` from `handle_telegram_event`)
- `text` (required) — Message text. Supports Markdown formatting (bold, italic, code, links)
- `parse_mode` (optional, default `Markdown`) — `Markdown`, `MarkdownV2`, or `HTML`
- `reply_to_message_id` (optional) — Reply to a specific message by its ID

**Use when:** Replying to a user's Telegram message, sending a proactive notification, or any time you need to send text to a Telegram chat.

---

### `telegram_send_photo`
Send a photo to a Telegram chat using a public image URL.

**Parameters:**
- `chat_id` (required) — Telegram chat ID
- `photo_url` (required) — Public URL of the image
- `caption` (optional) — Caption text. Supports Markdown

**Use when:** The user asks you to send an image, or you want to include a visual with your response.

---

### `telegram_get_chat`
Get information about a Telegram chat or user.

**Parameters:**
- `chat_id` (required) — Telegram chat ID or @username

**Returns:** Chat type (private/group/supergroup/channel), name, username, and member count.

**Use when:** You need to look up details about a chat before sending a message, or the user asks about a specific chat or user.

---

## Handling Incoming Messages

When a Telegram message arrives, you will receive an instruction like:

> "A Telegram message has arrived. Call the `handle_telegram_event` tool now with the following payload..."

**Always follow this flow:**

```
1. handle_telegram_event payload={...}   ← extract chat_id, sender, text
2. Decide: reply, ignore, or take action
3. telegram_send_message chat_id=... text="..."   ← reply to the user
```

**Decision rules:**
- If the message contains a question or request → answer it and reply via `telegram_send_message`
- If the message is a greeting → greet back warmly
- If the message is from a disallowed chat → the tool will return an "ignored" result; do not reply
- If the update has no text (photo, sticker, voice without caption) → acknowledge with a brief reply unless the user has standing instructions to ignore media

**Reply style:**
- Keep replies concise — Telegram is a messaging app, not a document editor
- Use Markdown sparingly: **bold** for emphasis, `code` for technical values
- Do not use MarkdownV2 unless you need to escape special characters — it's error-prone
- For multi-step answers, use short paragraphs rather than long bullet lists

---

## Sending Proactive Messages

If the user asks you to send a Telegram message (e.g. "message me on Telegram when the task is done"):

1. Use `telegram_send_message` with the user's known `chat_id`
2. If you don't know the chat ID, ask the user — they can find it by messaging `@userinfobot` on Telegram

---

## Example Workflows

**Incoming message: "What's the weather today?"**
```
1. handle_telegram_event payload={...}
   → chat_id="123456789", sender_name="Alice", text="What's the weather today?"
2. [look up weather or use knowledge]
3. telegram_send_message chat_id="123456789" text="It's 22°C and sunny in your area today ☀️"
```

**Incoming message: "Remind me to call Bob at 3pm"**
```
1. handle_telegram_event payload={...}
   → chat_id="123456789", text="Remind me to call Bob at 3pm"
2. schedule_reminder ... (set the reminder)
3. telegram_send_message chat_id="123456789" text="Done! I'll remind you to call Bob at 3:00 PM."
```

**User asks: "Send me a photo of a sunset"**
```
1. [find or generate a public image URL]
2. telegram_send_photo chat_id="123456789" photo_url="https://..." caption="Here's a sunset 🌅"
```

**Proactive notification after a background task:**
```
1. [task completes]
2. telegram_send_message chat_id="123456789" text="✅ Your report is ready."
```

---

## Error Handling

- If `telegram_send_message` fails with a 403 error → the bot was blocked by the user; inform them in the next available channel
- If `telegram_send_message` fails with a 400 error → the chat ID may be wrong; ask the user to confirm it
- If `handle_telegram_event` returns "not in allowed_chat_ids" → do not reply; this is a security control
- If the bot token is missing → the plugin is not configured; tell the user to add their bot token in plugin settings
