# Spotify Plugin

Control music playback, search for tracks, and view listening history.

## Core Workflow

Playing a specific song requires searching first to get the Spotify URI, then queuing it:

```
"Play [song]"  →  search_spotify → queue_track → skip_track (to jump to it)
"What's playing?" → now_playing
"Pause" / "Resume" → play_pause
"Next" / "Skip" → skip_track
"What did I listen to?" → recent_tracks
```

## Decision Rules

**Playing a specific song or artist:**
- Search first with `search_spotify` to get the Spotify URI, then `queue_track` to add it, then `skip_track` to jump to it immediately.
- Be specific in search queries: "Blinding Lights The Weeknd" works better than just "Weeknd".
- Show the user what you found before queuing if the match might not be exact.

**Resuming playback:**
- If the user says "play music" or "play something" without specifying a track, check `now_playing` first. If something is already loaded, just resume with `play_pause action="play"`.

**Checking state:**
- Call `now_playing` when you're unsure of the current playback state before issuing play/pause commands.

**Presenting results:**
- Be natural: "Now playing: Blinding Lights by The Weeknd (2:34 / 3:22)"
- For search results, show track name and artist. Skip album unless relevant.
- For `recent_tracks`, group by day if showing many results.

**Error handling:**
- 403 error → Spotify Premium is required for playback control. Tell the user.
- 404 error → No active device. Tell the user: "Open Spotify on a device first, then I can control it."
- No search results → Try a broader search or ask the user to clarify.

## Rate Limits

| Quota | Limit |
|---|---|
| Requests per 30 seconds | ~180 (rolling window) |

Spotify is lenient. Playback control has no practical limit. If you get a 429, wait the `Retry-After` seconds before retrying.

## Example Workflows

**User: "Play Daft Punk"**
```
1. search_spotify query="Daft Punk" type="track" limit=5
2. queue_track uri="spotify:track:..." (top result)
3. skip_track → jump to it immediately
4. "Now playing: Get Lucky by Daft Punk"
```

**User: "What's playing?"**
```
1. now_playing
2. "You're listening to Blinding Lights by The Weeknd (2:34 / 3:22)"
```

**User: "What did I listen to earlier?"**
```
1. recent_tracks limit=10
2. "Here's what you've been listening to today:
   - Get Lucky — Daft Punk
   - Blinding Lights — The Weeknd
   - Starboy — The Weeknd
   ..."
```
