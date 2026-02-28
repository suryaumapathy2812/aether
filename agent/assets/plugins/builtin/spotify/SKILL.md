# Spotify Plugin

You have access to Spotify tools for controlling music playback and browsing the user's library.

---

## Tools Available

### `now_playing`
Get the currently playing track.

**Parameters:** None

**Returns:** Track name, artist, album, playback progress, duration, shuffle/repeat state, and whether playback is active.

**Use when:** The user asks "what's playing?", "what song is this?", or before controlling playback to understand the current state.

---

### `play_pause`
Play or pause playback on the active device.

**Parameters:**
- `action` (required) — `"play"` or `"pause"`

**Use when:** The user says "play", "pause", "resume", "stop the music", or similar.

---

### `skip_track`
Skip to the next or previous track.

**Parameters:**
- `direction` (required) — `"next"` or `"previous"`

**Use when:** The user says "next song", "skip this", "go back", "previous track".

---

### `search_spotify`
Search for tracks, artists, or albums on Spotify.

**Parameters:**
- `query` (required) — Search term (track name, artist, album, or combination)
- `type` (optional, default `"track"`) — What to search: `"track"`, `"artist"`, `"album"`
- `limit` (optional, default `5`) — Number of results to return

**Returns:** List of results with name, artist/album, Spotify URI, and popularity score.

**Use when:** The user wants to play something specific, or you need a Spotify URI before queuing a track.

---

### `queue_track`
Add a track to the playback queue.

**Parameters:**
- `uri` (required) — Spotify URI from search results (e.g. `spotify:track:4iV5W9uYEdYUVa79Axb7Rh`)

**Use when:** The user wants to play or queue a specific track found via `search_spotify`.

---

### `recent_tracks`
Get the user's recently played tracks.

**Parameters:**
- `limit` (optional, default `10`) — Number of recent tracks to return

**Returns:** List of recently played tracks with name, artist, and when they were played.

**Use when:** The user asks "what did I listen to recently?" or "what was that song I played earlier?".

---

## Decision Rules

**Playing music:**
- For "play [song/artist]": `search_spotify` → `queue_track` → optionally `skip_track next` to jump to it immediately
- For "play something" or "play music": check `now_playing` first — if already playing, just resume with `play_pause action="play"`
- Don't queue tracks without the user asking — only queue when they explicitly want to play something

**Controlling playback:**
- Always check `now_playing` first if you're unsure of the current state before issuing play/pause
- For "skip" or "next" → `skip_track direction="next"`
- For "go back" or "previous" → `skip_track direction="previous"`

**Searching:**
- Be specific in search queries: "Blinding Lights The Weeknd" beats just "Weeknd"
- If the user says "play something by [artist]", search for the artist and queue their top track
- Show the user what you found before queuing, especially if the match might not be exact

**Presenting results:**
- Keep it natural: "Now playing: Blinding Lights by The Weeknd" not raw JSON
- For search results, show track name and artist — skip album unless relevant
- For `recent_tracks`, group by day if showing many results

**Error handling:**
- **403 error** → Spotify Premium is required for playback control. Tell the user: "Playback control requires Spotify Premium."
- **404 error** → No active device. Tell the user: "Open Spotify on a device first, then I can control it."
- **No results** → Try a broader search or ask the user to clarify the song/artist name

---

## Example Workflows

**"What's playing?"**
```
1. now_playing
2. "You're listening to Blinding Lights by The Weeknd (2:34 / 3:22)"
```

**"Play Daft Punk"**
```
1. search_spotify query="Daft Punk" type="artist"
2. search_spotify query="Daft Punk top tracks" type="track"
3. queue_track uri="spotify:track:..."
4. skip_track direction="next"  (to jump to it immediately)
5. "Now queuing: Get Lucky by Daft Punk"
```

**"Skip this song"**
```
1. skip_track direction="next"
2. now_playing  (optional, to confirm what's playing now)
3. "Skipped! Now playing: [next track]"
```

**"What did I listen to earlier?"**
```
1. recent_tracks limit=10
2. "Here's what you've been listening to: [list]"
```
