---
name: forms
description: Google Forms API — create forms, add questions, collect responses
integration: google-workspace
---
# Google Forms API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://forms.googleapis.com/v1/forms`
- Token auto-refreshes on 401 response

## Create Form

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "info": {
      "title": "My Survey",
      "documentTitle": "My Survey"
    }
  }' \
  "https://forms.googleapis.com/v1/forms"
```

Returns the created form with `formId`.

## Get Form

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://forms.googleapis.com/v1/forms/{FORM_ID}"
```

## Batch Update (Add Questions)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "createItem": {
          "item": {
            "title": "What is your name?",
            "questionItem": {
              "question": {
                "required": true,
                "textQuestion": {
                  "paragraph": false
                }
              }
            }
          },
          "location": {"index": 0}
        }
      }
    ]
  }' \
  "https://forms.googleapis.com/v1/forms/{FORM_ID}:batchUpdate"
```

### Question Types

**Short Answer:**
```json
"textQuestion": {"paragraph": false}
```

**Paragraph:**
```json
"textQuestion": {"paragraph": true}
```

**Multiple Choice:**
```json
"choiceQuestion": {
  "type": "RADIO",
  "options": [
    {"value": "Option A"},
    {"value": "Option B"}
  ]
}
```

**Checkbox (multiple select):**
```json
"choiceQuestion": {
  "type": "CHECKBOX",
  "options": [
    {"value": "Choice 1"},
    {"value": "Choice 2"}
  ]
}
```

**Dropdown:**
```json
"choiceQuestion": {
  "type": "DROP_DOWN",
  "options": [
    {"value": "Option 1"},
    {"value": "Option 2"}
  ]
}
```

## Get Form Responses

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://forms.googleapis.com/v1/forms/{FORM_ID}/responses"
```

### Query Parameters
- `filter` — filter responses (e.g., `timestamp > 2025-01-01T00:00:00Z`)

## Get Individual Response

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://forms.googleapis.com/v1/forms/{FORM_ID}/responses/{RESPONSE_ID}"
```

## Set Publish Settings

```bash
curl -s -X PATCH \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "publishSettings": {
      "publishState": "PUBLISHED",
      "isPublishedToWeb": true
    }
  }' \
  "https://forms.googleapis.com/v1/forms/{FORM_ID}/setPublishSettings"
```

## Rate Limits
- 300 requests per minute per project

## Error Handling
- **400 Bad Request**: Invalid question structure
- **401 Unauthorized**: Token expired — auto-refreshes, retry
- **403 Forbidden**: Quota exceeded or insufficient permissions
- **404 Not Found**: Form ID doesn't exist
- **429 Rate Limited**: Wait and retry
