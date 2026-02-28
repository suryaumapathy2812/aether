"""Spotify tools for playback control and music browsing.

Provides tools to interact with Spotify Web API:
- Now playing (current track info)
- Play / pause toggle
- Skip to next track
- Search tracks, artists, albums
- Queue a track
- Recent listening history

All tools receive credentials at call time via ``self._context``
(set by ``safe_execute`` from the plugin context store).
No ``__init__`` args required — the loader can instantiate with ``cls()``.
"""

from __future__ import annotations

import logging

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult
from aether.tools.refresh_oauth_token import RefreshOAuthTokenTool

logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"


class _SpotifyTool(AetherTool):
    """Base for Spotify tools — provides token extraction from runtime context."""

    def _get_token(self) -> str | None:
        ctx = getattr(self, "_context", None)
        return ctx.get("access_token") if ctx else None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}


class NowPlayingTool(_SpotifyTool):
    """Get the currently playing track."""

    name = "now_playing"
    description = "Get the currently playing track on Spotify"
    status_text = "Checking what's playing..."
    parameters = []

    async def execute(self, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Spotify not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{SPOTIFY_API}/me/player/currently-playing",
                    headers=self._auth_headers(),
                )

                if resp.status_code == 204 or not resp.content:
                    return ToolResult.success("Nothing is currently playing.")

                resp.raise_for_status()
                data = resp.json()

            if not data.get("item"):
                return ToolResult.success("Nothing is currently playing.")

            item = data["item"]
            track = item.get("name", "Unknown")
            artists = ", ".join(a["name"] for a in item.get("artists", []))
            album = item.get("album", {}).get("name", "")
            progress_ms = data.get("progress_ms", 0)
            duration_ms = item.get("duration_ms", 0)
            is_playing = data.get("is_playing", False)

            progress = f"{progress_ms // 60000}:{(progress_ms // 1000) % 60:02d}"
            duration = f"{duration_ms // 60000}:{(duration_ms // 1000) % 60:02d}"
            state = "Playing" if is_playing else "Paused"

            output = f"**{state}:** {track}\n"
            output += f"**Artist:** {artists}\n"
            if album:
                output += f"**Album:** {album}\n"
            output += f"**Progress:** {progress} / {duration}"

            return ToolResult.success(
                output, track=track, artist=artists, is_playing=is_playing
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return ToolResult.fail("Spotify Premium required for playback info.")
            logger.error(f"Spotify API error: {e}", exc_info=True)
            return ToolResult.fail(f"Spotify API error: {e}")
        except Exception as e:
            logger.error(f"Error getting now playing: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class PlayPauseTool(_SpotifyTool):
    """Toggle play/pause on the active Spotify device."""

    name = "play_pause"
    description = "Play or pause Spotify playback"
    status_text = "Toggling playback..."
    parameters = [
        ToolParam(
            name="action",
            type="string",
            description="'play' or 'pause'",
            required=True,
            enum=["play", "pause"],
        ),
    ]

    async def execute(self, action: str = "play", **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Spotify not connected — missing access token.")

        try:
            endpoint = "play" if action == "play" else "pause"
            method = "PUT" if action == "play" else "PUT"

            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method,
                    f"{SPOTIFY_API}/me/player/{endpoint}",
                    headers=self._auth_headers(),
                )

                if resp.status_code == 404:
                    return ToolResult.fail(
                        "No active Spotify device found. "
                        "Open Spotify on a device first."
                    )
                resp.raise_for_status()

            return ToolResult.success(
                f"Playback {'resumed' if action == 'play' else 'paused'}."
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return ToolResult.fail("Spotify Premium required for playback control.")
            logger.error(f"Spotify API error: {e}", exc_info=True)
            return ToolResult.fail(f"Spotify API error: {e}")
        except Exception as e:
            logger.error(f"Error toggling playback: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class SkipTrackTool(_SpotifyTool):
    """Skip to the next or previous track."""

    name = "skip_track"
    description = "Skip to the next or previous track on Spotify"
    status_text = "Skipping..."
    parameters = [
        ToolParam(
            name="direction",
            type="string",
            description="'next' or 'previous'",
            required=False,
            default="next",
            enum=["next", "previous"],
        ),
    ]

    async def execute(self, direction: str = "next", **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Spotify not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{SPOTIFY_API}/me/player/{direction}",
                    headers=self._auth_headers(),
                )

                if resp.status_code == 404:
                    return ToolResult.fail(
                        "No active Spotify device found. "
                        "Open Spotify on a device first."
                    )
                resp.raise_for_status()

            return ToolResult.success(
                f"Skipped to {'next' if direction == 'next' else 'previous'} track."
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return ToolResult.fail("Spotify Premium required for playback control.")
            logger.error(f"Spotify API error: {e}", exc_info=True)
            return ToolResult.fail(f"Spotify API error: {e}")
        except Exception as e:
            logger.error(f"Error skipping track: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class SearchSpotifyTool(_SpotifyTool):
    """Search Spotify for tracks, artists, or albums."""

    name = "search_spotify"
    description = "Search Spotify for tracks, artists, or albums"
    status_text = "Searching Spotify..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description="Search query (song name, artist, etc.)",
            required=True,
        ),
        ToolParam(
            name="search_type",
            type="string",
            description="Type of search: 'track', 'artist', or 'album'",
            required=False,
            default="track",
            enum=["track", "artist", "album"],
        ),
        ToolParam(
            name="limit",
            type="integer",
            description="Max results to return",
            required=False,
            default=5,
        ),
    ]

    async def execute(
        self, query: str, search_type: str = "track", limit: int = 5, **_
    ) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Spotify not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{SPOTIFY_API}/search",
                    headers=self._auth_headers(),
                    params={"q": query, "type": search_type, "limit": limit},
                )
                resp.raise_for_status()
                data = resp.json()

            key = f"{search_type}s"  # tracks, artists, albums
            items = data.get(key, {}).get("items", [])

            if not items:
                return ToolResult.success(f"No {search_type}s found for '{query}'.")

            output = f"**Search results for '{query}':**\n"

            for i, item in enumerate(items, 1):
                if search_type == "track":
                    artists = ", ".join(a["name"] for a in item.get("artists", []))
                    album = item.get("album", {}).get("name", "")
                    uri = item.get("uri", "")
                    output += f"\n{i}. **{item['name']}** by {artists}\n"
                    if album:
                        output += f"   Album: {album}\n"
                    output += f"   URI: {uri}\n"
                elif search_type == "artist":
                    genres = ", ".join(item.get("genres", [])[:3]) or "N/A"
                    followers = item.get("followers", {}).get("total", 0)
                    output += f"\n{i}. **{item['name']}**\n"
                    output += f"   Genres: {genres}\n"
                    output += f"   Followers: {followers:,}\n"
                elif search_type == "album":
                    artists = ", ".join(a["name"] for a in item.get("artists", []))
                    year = item.get("release_date", "")[:4]
                    tracks = item.get("total_tracks", 0)
                    uri = item.get("uri", "")
                    output += f"\n{i}. **{item['name']}** by {artists}\n"
                    output += f"   Year: {year} | Tracks: {tracks}\n"
                    output += f"   URI: {uri}\n"

            return ToolResult.success(output, count=len(items))

        except Exception as e:
            logger.error(f"Error searching Spotify: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class QueueTrackTool(_SpotifyTool):
    """Add a track to the playback queue."""

    name = "queue_track"
    description = "Add a track to the Spotify playback queue"
    status_text = "Adding to queue..."
    parameters = [
        ToolParam(
            name="uri",
            type="string",
            description="Spotify track URI (e.g. spotify:track:4iV5W9uYEdYUVa79Axb7Rh)",
            required=True,
        ),
    ]

    async def execute(self, uri: str, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Spotify not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{SPOTIFY_API}/me/player/queue",
                    headers=self._auth_headers(),
                    params={"uri": uri},
                )

                if resp.status_code == 404:
                    return ToolResult.fail(
                        "No active Spotify device found. "
                        "Open Spotify on a device first."
                    )
                resp.raise_for_status()

            return ToolResult.success(f"Track added to queue: {uri}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return ToolResult.fail("Spotify Premium required for queue control.")
            logger.error(f"Spotify API error: {e}", exc_info=True)
            return ToolResult.fail(f"Spotify API error: {e}")
        except Exception as e:
            logger.error(f"Error queuing track: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RecentTracksTool(_SpotifyTool):
    """Get recently played tracks."""

    name = "recent_tracks"
    description = "Get your recently played tracks on Spotify"
    status_text = "Checking listening history..."
    parameters = [
        ToolParam(
            name="limit",
            type="integer",
            description="Max tracks to return",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, limit: int = 10, **_) -> ToolResult:
        if not self._get_token():
            return ToolResult.fail("Spotify not connected — missing access token.")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{SPOTIFY_API}/me/player/recently-played",
                    headers=self._auth_headers(),
                    params={"limit": limit},
                )
                resp.raise_for_status()
                data = resp.json()

            items = data.get("items", [])
            if not items:
                return ToolResult.success("No recent listening history.")

            output = "**Recently Played:**\n"
            for i, item in enumerate(items, 1):
                track = item.get("track", {})
                name = track.get("name", "Unknown")
                artists = ", ".join(a["name"] for a in track.get("artists", []))
                played_at = item.get("played_at", "")
                # Format timestamp: "2026-02-18T07:15:00.000Z" → "Feb 18, 07:15"
                if played_at:
                    played_at = played_at.replace("T", " ").split(".")[0]

                output += f"\n{i}. **{name}** by {artists}\n"
                if played_at:
                    output += f"   Played: {played_at}\n"

            return ToolResult.success(output, count=len(items))

        except Exception as e:
            logger.error(f"Error getting recent tracks: {e}", exc_info=True)
            return ToolResult.fail(f"Error: {e}")


class RefreshSpotifyTokenTool(RefreshOAuthTokenTool):
    """Refresh the Spotify OAuth access token before it expires.

    Called automatically by the cron system every 50 minutes.
    Can also be called manually if Spotify tools start returning auth errors.
    """

    name = "refresh_spotify_token"
    plugin_name = "spotify"
    description = (
        "Refresh the Spotify OAuth access token. "
        "Call this when Spotify tools return authentication errors, "
        "or when instructed by the system to prevent token expiry."
    )
