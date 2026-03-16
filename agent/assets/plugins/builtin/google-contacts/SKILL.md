# Google Contacts Plugin

Search and look up contact information from the user's Google Contacts.

## Core Workflow

Contact lookups are usually one or two steps:

```
search_contacts → present result
                → get_contact (only if more detail needed)
```

`search_contacts` handles name, email, and phone lookups and returns the most common fields (name, emails, phones, organization). Use `get_contact` only when you need fields not in the search result — like birthday, full address, or notes.

## Decision Rules

**Looking up contacts:**
- If multiple contacts match (e.g., searching "John"), list them and ask which one the user means.
- If no contacts match, say "I couldn't find anyone named [X] in your contacts" and suggest trying a different spelling.

**Presenting contact info:**
- Be natural: "Priya's email is priya@example.com" — not raw field names.
- Only share the specific detail the user asked for. Don't dump the entire contact record.
- If a contact has multiple emails (work/personal), mention both and ask which to use.

**Privacy:**
- Don't volunteer contact details the user didn't ask for. If they ask for a phone number, give just the phone number.

**Cross-plugin integration:**
- When the user says "email Priya", use `search_contacts` to find Priya's email address, then hand off to the Gmail plugin to compose.
- When the user says "call Priya", look up her phone number and present it.

## Rate Limits

| Quota | Limit |
|---|---|
| Read requests per minute per user | 90 |
| Concurrent requests | 10 per user |

Contact API rate limits are tighter than other Google APIs. Avoid more than 5-10 parallel requests. A single lookup is usually one call.

## Example Workflows

**User: "What's Priya's email?"**
```
1. search_contacts query="Priya"
2. "Priya Sharma's email is priya.sharma@company.com"
   (If multiple Priyas: "I found 2 contacts named Priya — which one?")
```

**User: "Email Rahul about the project update"**
```
1. search_contacts query="Rahul" → get email address
2. Hand off to Gmail plugin to compose the email to rahul@example.com
```

**User: "What's the phone number for Acme Corp?"**
```
1. search_contacts query="Acme Corp"
2. "Acme Corp's main number is +1-555-0100 (contact: John Smith, Sales Director)"
```
