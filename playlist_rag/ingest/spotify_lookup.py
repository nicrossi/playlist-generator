"""Resolve real Spotify track_ids for fresh tracks via the Spotify Web API.

Spotify deprecated the audio-features endpoint (Nov 2024), but the search
endpoint still returns track identity + metadata. We use the client-credentials
flow (no user login) to attach a canonical track_id to locally-extracted tracks
so they share a primary key with the Kaggle catalog.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests
from rapidfuzz import fuzz

from playlist_rag.config import settings
from unify import canonicalize, strip_title_suffixes

logger = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"

# Artist and title are scored separately (token_set_ratio handles subsets and
# punctuation better than one combined string). A hit must clear both.
ARTIST_MATCH = 85
TITLE_MATCH = 70

# Multi-artist separators seen across datasets (", " | " & " | " x " | "feat").
_ARTIST_SPLIT_RE = re.compile(r"\s*(?:,|&|/| x | feat\.?| ft\.?| con | with )\s*", re.I)


def primary_artist(artists_field: str) -> str:
    """First artist from a multi-artist field ('Yng Lvcas & Peso Pluma' -> 'Yng Lvcas')."""
    if not isinstance(artists_field, str) or not artists_field:
        return ""
    return _ARTIST_SPLIT_RE.split(artists_field, maxsplit=1)[0].strip()


class SpotifyAuthError(RuntimeError):
    """Raised when client credentials are missing or rejected."""


@dataclass
class SpotifyMatch:
    track_id: str
    track_name: str
    track_artist: str
    track_album_name: str
    track_album_id: str
    track_album_release_date: str
    track_popularity: int
    duration_ms: int


class SpotifyClient:
    """Thin client-credentials wrapper around the Spotify search endpoint."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.client_id = client_id or settings.spotify_client_id
        self.client_secret = client_secret or settings.spotify_client_secret
        if not self.client_id or not self.client_secret:
            raise SpotifyAuthError(
                "set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (.env) "
                "or run with --no-spotify"
            )
        self._session = session or requests.Session()
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    def _get_token(self) -> str:
        if self._token and time.monotonic() < self._token_expiry:
            return self._token
        resp = self._session.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=15,
        )
        if resp.status_code != 200:
            raise SpotifyAuthError(f"token request failed: {resp.status_code} {resp.text}")
        payload = resp.json()
        self._token = payload["access_token"]
        # Refresh a minute early to dodge clock skew near expiry.
        self._token_expiry = time.monotonic() + payload.get("expires_in", 3600) - 60
        return self._token

    def search_track(self, artist: str, title: str) -> Optional[SpotifyMatch]:
        """Return the best track match, or None if nothing clears both thresholds."""
        artist = primary_artist(artist) or artist
        clean_title = strip_title_suffixes(title)
        query = f"track:{clean_title} artist:{artist}"
        resp = self._session.get(
            SEARCH_URL,
            headers={"Authorization": f"Bearer {self._get_token()}"},
            params={"q": query, "type": "track", "limit": 5},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("Spotify search failed (%s) for %r", resp.status_code, query)
            return None

        want_artist = canonicalize(artist)
        want_title = canonicalize(clean_title)
        best: Optional[SpotifyMatch] = None
        best_score = -1.0
        for item in resp.json().get("tracks", {}).get("items", []):
            got_artist = item["artists"][0]["name"] if item.get("artists") else ""
            a = fuzz.token_set_ratio(want_artist, canonicalize(got_artist))
            t = fuzz.token_set_ratio(want_title, canonicalize(strip_title_suffixes(item["name"])))
            if a >= ARTIST_MATCH and t >= TITLE_MATCH and a + t > best_score:
                best_score = a + t
                best = _to_match(item, got_artist)
        if best is None:
            logger.info("No Spotify match for %s - %s", artist, title)
        return best


def _to_match(item: dict, artist: str) -> SpotifyMatch:
    album = item.get("album") or {}
    return SpotifyMatch(
        track_id=item["id"],
        track_name=item["name"],
        track_artist=artist,
        track_album_name=album.get("name", ""),
        track_album_id=album.get("id", ""),
        track_album_release_date=album.get("release_date", ""),
        track_popularity=item.get("popularity", 0),
        duration_ms=item.get("duration_ms", 0),
    )
