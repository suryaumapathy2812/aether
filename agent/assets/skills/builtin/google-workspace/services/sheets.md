---
name: sheets
description: Google Sheets API — create, read, update spreadsheets and values
integration: google-workspace
---
# Google Sheets API

## Authentication
- **Env var**: `$GOOGLE_WORKSPACE_ACCESS_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["google-workspace"]` to the execute tool
- **Base URL**: `https://sheets.googleapis.com/v4/spreadsheets`
- Token auto-refreshes on 401 response

## Create Spreadsheet

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "properties": {"title": "My Spreadsheet"},
    "sheets": [{"properties": {"title": "Sheet1"}}]
  }' \
  "https://sheets.googleapis.com/v4/spreadsheets"
```

## Get Spreadsheet

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}"
```

## Read Values

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{RANGE}"
```

- `{RANGE}` examples: `Sheet1!A1:D10`, `Sheet1`, `Sheet1!A:A` (column A)

## Update Values

```bash
curl -s -X PUT \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "values": [["Name", "Age"], ["Alice", 30], ["Bob", 25]],
    "majorDimension": "ROWS"
  }' \
  "https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{RANGE}?valueInputOption=USER_ENTERED"
```

### valueInputOption
- `RAW` — values stored as-is (no parsing)
- `USER_ENTERED` — values parsed as if typed by user (dates, formulas, etc.)

## Append Values

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "values": [["Charlie", 35], ["Diana", 28]]
  }' \
  "https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{RANGE}:append?valueInputOption=USER_ENTERED"
```

## Batch Update (Formatting, Adding Sheets, etc.)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "addSheet": {
          "properties": {"title": "New Sheet"}
        }
      }
    ]
  }' \
  "https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}:batchUpdate"
```

## Batch Get Values

```bash
curl -s -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  "https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values:batchGet?ranges=Sheet1!A1:B5&ranges=Sheet2!C1:D5"
```

## Clear Values

```bash
curl -s -X POST \
  -H "Authorization: Bearer $GOOGLE_WORKSPACE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' \
  "https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{RANGE}:clear"
```

## Rate Limits
- 300 requests per minute per project
- 60 requests per minute per user per project

## Error Handling
- **400 Bad Request**: Invalid range or request body
- **401 Unauthorized**: Token expired — auto-refreshes, retry
- **403 Forbidden**: Quota exceeded or insufficient permissions
- **404 Not Found**: Spreadsheet ID doesn't exist
- **429 Rate Limited**: Wait and retry
