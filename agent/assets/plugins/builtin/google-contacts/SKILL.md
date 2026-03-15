# Google Contacts Plugin

You have access to Google Contacts tools for searching and looking up contact information.

---

## Tools Available

### `search_contacts`
Search contacts by name, email address, or phone number.

**Parameters:**
- `query` (required) — Name, email, or phone number to search for

**Returns:** Matching contacts with name, email addresses, phone numbers, organization, and resource name.

**Use when:** The user asks for someone's contact details, email address, or phone number. Also use this before sending an email to verify the correct address.

---

### `get_contact`
Get full details of a specific contact by their resource name.

**Parameters:**
- `resource_name` (required) — The contact's resource name from `search_contacts` (e.g. `people/c12345`)

**Returns:** Complete contact record including all emails, phones, addresses, birthday, notes, and organization.

**Use when:** `search_contacts` returned a contact but you need more detail (e.g. multiple email addresses, their job title, or notes).

---

## Pagination & Limits

**Contacts API default page size:** 10 results per search (max 30).

**For broad searches (common names like "John"):**
1. Set `page_size` higher (up to 30) to get more results in one call
2. The response includes a `nextPageToken` if more contacts match
3. If the user needs a comprehensive list, make follow-up calls with `pageToken`

---

## Rate Limits

| Quota | Limit |
|---|---|
| Read requests per minute per user | 90 |
| Read requests per day per user | 20,000 |
| Concurrent requests | 10 per user |

**In practice:** Contact search rate limits are tighter than other Google APIs. Avoid making more than 5-10 parallel requests. For a single lookup, one call is usually enough.

---

## Decision Rules

**Looking up contacts:**
- Use `search_contacts` first — it handles name, email, and phone lookups
- If multiple contacts match, list them and ask the user which one they mean
- Use `get_contact` only when you need fields not returned by search (e.g. birthday, full address, notes)

**Presenting contact info:**
- Be natural: "Priya's email is priya@example.com" not "email_addresses[0].value: priya@example.com"
- Only share the specific detail the user asked for — don't dump the entire contact record
- If a contact has multiple emails (work/personal), mention both and ask which to use

**Privacy:**
- Don't volunteer contact details the user didn't ask for
- If the user asks for someone's phone number, give just the phone number — not their full profile

**Integration with Gmail:**
- When the user says "email Priya", use `search_contacts` to find Priya's email address before composing
- When the user says "call Priya", use `search_contacts` to find her phone number

**Error handling:**
- If no contacts match, say "I couldn't find anyone named [X] in your contacts" and ask if they want to try a different spelling
- If the query is ambiguous (e.g. "John"), list all matches and ask which one

---

## Example Workflows

**"What's Priya's email?"**
```
1. search_contacts query="Priya"
2. "Priya Sharma's email is priya.sharma@company.com"
   (If multiple Priyas: "I found 2 contacts named Priya — which one do you mean?")
```

**"Email Rahul about the project update"**
```
1. search_contacts query="Rahul" → get email address
2. Proceed to compose email using Gmail plugin
```

**"What's the phone number for Acme Corp?"**
```
1. search_contacts query="Acme Corp"
2. "Acme Corp's main number is +1-555-0100 (contact: John Smith)"
```
