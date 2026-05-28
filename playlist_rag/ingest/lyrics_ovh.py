from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from playlist_rag.indexing.normalize import clean_lyrics
from unify import extract_primary_artist, strip_title_suffixes

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.lyrics.ovh/v1"
_CACHE_FOUND = "found"
_CACHE_MISS = "not_found"


@dataclass
class FetchResult:
    lyrics: Optional[str] = None
    status: str = _CACHE_MISS
    matched_title: Optional[str] = None
    http_status: Optional[int] = None
    from_cache: bool = False


@dataclass
class BackfillStats:
    candidates: int = 0
    skipped_has_lyrics: int = 0
    cache_hits: int = 0
    fetched: int = 0
    found: int = 0
    not_found: int = 0
    errors: int = 0
    backfilled_ids: list[str] = field(default_factory=list)


def _is_blank_lyrics(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and str(value) == "nan":
        return True
    text = str(value).strip()
    return not text or text.lower() in ("nan", "none", "null")


def title_variants(track_name: str) -> list[str]:
    if not track_name:
        return []
    stripped = strip_title_suffixes(track_name).strip()
    variants: list[str] = []
    for candidate in (stripped, track_name.strip()):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _cache_path(cache_dir: Path, spotify_track_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", spotify_track_id)[:120]
    return cache_dir / f"{safe}.json"


def read_cache(cache_dir: Path, spotify_track_id: str) -> Optional[FetchResult]:
    path = _cache_path(cache_dir, spotify_track_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FetchResult(
            lyrics=data.get("lyrics"),
            status=data.get("status", _CACHE_MISS),
            matched_title=data.get("matched_title"),
            http_status=data.get("http_status"),
        )
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Invalid cache %s: %s", path, e)
        return None


def write_cache(cache_dir: Path, spotify_track_id: str, result: FetchResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, spotify_track_id)
    payload = {
        "spotify_track_id": spotify_track_id,
        "lyrics": result.lyrics,
        "status": result.status,
        "matched_title": result.matched_title,
        "http_status": result.http_status,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def fetch_lyrics(
    artist: str,
    title: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = 20.0,
    session: Optional[requests.Session] = None,
) -> FetchResult:
    sess = session or requests.Session()
    artist_slug = quote(artist.strip(), safe="")
    title_slug = quote(title.strip(), safe="")
    url = f"{api_base.rstrip('/')}/{artist_slug}/{title_slug}"
    try:
        resp = sess.get(url, timeout=timeout)
    except requests.RequestException as e:
        logger.debug("Request failed %s — %s: %s", artist, title, e)
        return FetchResult(status="error", http_status=None)

    if resp.status_code == 404:
        return FetchResult(status=_CACHE_MISS, http_status=404)
    if resp.status_code != 200:
        return FetchResult(status="error", http_status=resp.status_code)

    try:
        data = resp.json()
    except ValueError:
        return FetchResult(status="error", http_status=resp.status_code)

    raw = data.get("lyrics") if isinstance(data, dict) else None
    cleaned = clean_lyrics(raw if isinstance(raw, str) else None)
    if not cleaned:
        return FetchResult(status=_CACHE_MISS, http_status=resp.status_code)

    return FetchResult(
        lyrics=cleaned,
        status=_CACHE_FOUND,
        matched_title=title,
        http_status=resp.status_code,
    )


def fetch_lyrics_with_fallbacks(
    track_artist: str,
    track_name: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = 20.0,
    session: Optional[requests.Session] = None,
) -> FetchResult:
    artist = extract_primary_artist(track_artist) or track_artist.strip()
    if not artist or not track_name:
        return FetchResult(status=_CACHE_MISS)

    sess = session or requests.Session()
    for title in title_variants(track_name):
        result = fetch_lyrics(
            artist, title, api_base=api_base, timeout=timeout, session=sess
        )
        if result.status == _CACHE_FOUND and result.lyrics:
            return result
    return FetchResult(status=_CACHE_MISS)


def fetch_or_cache(
    cache_dir: Path,
    spotify_track_id: str,
    track_artist: str,
    track_name: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    delay_seconds: float = 0.75,
    timeout: float = 20.0,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
) -> FetchResult:
    if use_cache:
        cached = read_cache(cache_dir, spotify_track_id)
        if cached is not None:
            cached.from_cache = True
            return cached

    result = fetch_lyrics_with_fallbacks(
        track_artist,
        track_name,
        api_base=api_base,
        timeout=timeout,
        session=session,
    )
    write_cache(cache_dir, spotify_track_id, result)
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return result
