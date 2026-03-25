# Google Contacts API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://people.googleapis.com/v1`
- Token auto-refreshes on 401 response

## Search Contacts

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://people.googleapis.com/v1/people:searchContacts?query=Priya&readMask=names,emailAddresses,phoneNumbers,organizations&pageSize=10"
```

### Query Parameter
- `query` — matches against name, email, phone number, and organization
- `readMask` — comma-separated fields to return (required)
- `pageSize` — max results (default 10, max 30)

### Available Read Mask Fields
- `names` — display name
- `emailAddresses` — email addresses
- `phoneNumbers` — phone numbers
- `organizations` — company/title
- `addresses` — physical addresses
- `birthdays` — birthday
- `photos` — profile photo URL
- `biographies` — notes/bio

## Get a Specific Contact

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://people.googleapis.com/v1/people/{RESOURCE_NAME}?personFields=names,emailAddresses,phoneNumbers,organizations,addresses,birthdays"
```

### Resource Name Format
Contact resource names look like `people/c1234567890`. Get this from search results.

## List All Contacts

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://people.googleapis.com/v1/people/me/connections?readMask=names,emailAddresses,phoneNumbers&pageSize=100"
```

### Pagination
Response includes `nextPageToken` when more contacts exist:
```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://people.googleapis.com/v1/people/me/connections?readMask=names,emailAddresses,phoneNumbers&pageSize=100&pageToken=NEXT_PAGE_TOKEN"
```

## Response Structure

### Search Response
```json
{
  "results": [
    {
      "person": {
        "resourceName": "people/c1234567890",
        "names": [{"displayName": "Priya Sharma", "givenName": "Priya"}],
        "emailAddresses": [{"value": "priya@example.com", "type": "work"}],
        "phoneNumbers": [{"value": "+1-555-0100", "type": "mobile"}],
        "organizations": [{"name": "Acme Corp", "title": "Engineer"}]
      }
    }
  ]
}
```

### Multiple Matches
When a search returns multiple contacts, iterate through `results[]` and present options to the user.

## List Contact Groups

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://people.googleapis.com/v1/contactGroups?pageSize=50"
```

## Get Contacts in a Group

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://people.googleapis.com/v1/contactGroups/{GROUP_ID}?maxMembers=100"
```

Then fetch each member's details with `people/{RESOURCE_NAME}`.

## Error Handling
- **401 Unauthorized**: Token expired — system auto-refreshes, retry the command
- **403 Forbidden**: Insufficient scopes — ensure `contacts.readonly` scope is granted
- **404 Not Found**: Contact resource name is invalid or contact was deleted
- **429 Rate Limited**: Contact API allows ~90 read requests/minute per user — wait and retry
- **Empty results**: Contact doesn't exist — suggest trying a different spelling or search term
- If search returns too many results, refine the query with more specific terms
