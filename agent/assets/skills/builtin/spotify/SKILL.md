---
name: spotify
description: Spotify Web API — playback control, search, playlists, and user library
integration: spotify
---
# Spotify API

## Authentication
- **Env var**: `$SPOTIFY_TOKEN` (auto-injected via execute tool)
- **Credentials**: Pass `credentials=["spotify"]` to the execute tool
- **Base URL**: `https://api.spotify.com/v1`
- Token auto-refreshes on 401 response

## Get Currently Playing
```bash
curl -s -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/currently-playing"
```
Returns `null` if nothing is playing. When active, response includes `item.name`, `item.artists[].name`, `item.album.name`.

## Get Playback State
```bash
curl -s -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player"
```
Returns device info, progress, shuffle/repeat state.

## Play / Resume
```bash
# Resume playback
curl -s -X PUT \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/play"

# Play specific track
curl -s -X PUT \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"uris": ["spotify:track:TRACK_ID"]}' \
  "https://api.spotify.com/v1/me/player/play"

# Play a playlist
curl -s -X PUT \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context_uri": "spotify:playlist:PLAYLIST_ID"}' \
  "https://api.spotify.com/v1/me/player/play"
```

## Pause
```bash
curl -s -X PUT \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/pause"
```

## Skip Track
```bash
# Next
curl -s -X POST \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/next"

# Previous
curl -s -X POST \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/previous"
```

## Set Volume
```bash
curl -s -X PUT \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/volume?volume_percent=50"
```

## Set Shuffle / Repeat
```bash
# Shuffle on/off
curl -s -X PUT \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/shuffle?state=true"

# Repeat: off, track, context
curl -s -X PUT \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/repeat?state=track"
```

## Search
```bash
curl -s -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/search?q=artist:Daft+Punk&type=track&limit=5"
```
Types: `track`, `artist`, `album`, `playlist`, `show`, `episode`

## Get User Playlists
```bash
curl -s -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/playlists?limit=20"
```

## Recently Played
```bash
curl -s -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/recently-played?limit=10"
```

## Transfer Playback to Device
```bash
curl -s -X PUT \
  -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"device_ids": ["DEVICE_ID"]}' \
  "https://api.spotify.com/v1/me/player"
```

## Get Available Devices
```bash
curl -s -H "Authorization: Bearer $SPOTIFY_TOKEN" \
  "https://api.spotify.com/v1/me/player/devices"
```

## Rate Limits
- ~180 requests per minute per user
- No official hard limit published

## Error Handling
- **401**: Token expired — auto-refreshes, retry
- **403**: Insufficient scope for this action
- **404**: Resource not found or no active device
- **429**: Rate limited — respect `Retry-After` header
