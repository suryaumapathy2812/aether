# Google Slides API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://slides.googleapis.com/v1/presentations`
- Token auto-refreshes on 401 response

## Create Presentation

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "My Presentation"}' \
  "https://slides.googleapis.com/v1/presentations"
```

## Get Presentation

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://slides.googleapis.com/v1/presentations/{PRESENTATION_ID}"
```

## Get Page (Slide)

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://slides.googleapis.com/v1/presentations/{PRESENTATION_ID}/pages/{PAGE_ID}"
```

## Batch Update

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "createSlide": {
          "insertionIndex": 1,
          "slideLayoutReference": {
            "predefinedLayout": "TITLE_AND_BODY"
          }
        }
      }
    ]
  }' \
  "https://slides.googleapis.com/v1/presentations/{PRESENTATION_ID}:batchUpdate"
```

### Common Request Types

**Create Slide:**
```json
{
  "createSlide": {
    "insertionIndex": 1,
    "slideLayoutReference": {
      "predefinedLayout": "TITLE_AND_BODY"
    }
  }
}
```

**Insert Text into Shape:**
```json
{
  "insertText": {
    "objectId": "SHAPE_ID",
    "text": "Slide title text"
  }
}
```

**Replace Text:**
```json
{
  "replaceAllText": {
    "containsText": {"text": "{{PLACEHOLDER}}", "matchCase": true},
    "replaceText": "Actual Value"
  }
}
```

**Create Shape:**
```json
{
  "createShape": {
    "shapeType": "TEXT_BOX",
    "elementProperties": {
      "pageObjectId": "PAGE_ID",
      "size": {
        "width": {"magnitude": 300, "unit": "PT"},
        "height": {"magnitude": 100, "unit": "PT"}
      },
      "transform": {
        "scaleX": 1, "scaleY": 1,
        "translateX": 100, "translateY": 100,
        "unit": "PT"
      }
    }
  }
}
```

### Predefined Layouts
- `TITLE` — title only
- `TITLE_AND_BODY` — title + body text
- `TITLE_ONLY` — title only (blank body)
- `SECTION_TITLE` — section header
- `SECTION_HEADER_AND_DESCRIPTION`
- `ONE_COLUMN_TEXT`
- `TWO_COLUMN_TEXT`
- `BLANK` — no placeholders

## Rate Limits
- 300 requests per minute per project

## Error Handling
- **400 Bad Request**: Invalid object ID or request structure
- **401 Unauthorized**: Token expired — auto-refreshes, retry
- **403 Forbidden**: Quota exceeded or insufficient permissions
- **404 Not Found**: Presentation or page ID doesn't exist
- **429 Rate Limited**: Wait and retry
