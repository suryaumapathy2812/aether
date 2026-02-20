# Vobiz Telephony Plugin

This plugin enables Aether to make and receive phone calls via Vobiz.

## Capabilities

- **Make outbound calls**: Call the user or any phone number
- **Receive inbound calls**: Handle incoming calls on your Vobiz number
- **Voice conversation**: Full voice AI experience over the phone

## Tools

### `make_phone_call`

Make a phone call to the user or a specified number.

**Parameters:**
- `to_number` (optional): Phone number to call in E.164 format (e.g., +919123456789)
- `greeting` (optional): Greeting to speak when the call connects
- `reason` (optional): Brief reason for the call

**Example usage:**
```
User: "Call me and remind me about my meeting"
Agent: Uses make_phone_call tool with greeting="Hi! This is Aether calling to remind you about your meeting."
```

### `get_user_phone_number`

Check if the user has a phone number configured.

## Configuration

Before using this plugin, configure the following in the plugin settings:

1. **Auth ID**: Your Vobiz Auth ID (from console.vobiz.ai)
2. **Auth Token**: Your Vobiz Auth Token
3. **Vobiz Phone Number**: Your Vobiz phone number (used as caller ID)
4. **Your Phone Number** (optional): Your personal phone number for quick calls

## Webhook Setup

For inbound calls, configure your Vobiz number's webhook to point to:
```
https://your-server.com/plugins/vobiz/webhook
```

## Usage Guidelines

- Use phone calls for urgent or time-sensitive matters
- Always provide a clear greeting when initiating calls
- The user can interrupt the AI during calls (barge-in is supported)
- Calls are stored in memory for context continuity
