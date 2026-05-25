import math
import re
import unicodedata
from typing import Any, Optional

from playlist_rag.schemas import NormalizedTrack

_ANNOTATION_RE = re.compile(r"\[.*?\]", re.DOTALL)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_NA_SENTINELS = {"nan", "none", "null", ""}


def clean_lyrics(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    text = unicodedata.normalize("NFKC", raw)
    text = _ANNOTATION_RE.sub("", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = text.strip()
    return text or None


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def _coerce_int(v: Any) -> Optional[int]:
    if _is_missing(v):
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def _coerce_float(v: Any) -> Optional[float]:
    if _is_missing(v):
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _coerce_str(v: Any) -> Optional[str]:
    if _is_missing(v):
        return None
    s = str(v).strip()
    if s.lower() in _NA_SENTINELS:
        return None
    return s


def _coerce_list(v: Any) -> list[str]:
    if _is_missing(v):
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    try:
        out = []
        for item in v:
            s = _coerce_str(item)
            if s:
                out.append(s)
        return out
    except TypeError:
        return []


def normalize_row(row: dict) -> NormalizedTrack:
    return NormalizedTrack(
        spotify_track_id=str(row["track_id"]),
        track_name=_coerce_str(row.get("track_name")) or "Unknown",
        track_artist=_coerce_str(row.get("track_artist")) or "Unknown",
        track_album_name=_coerce_str(row.get("track_album_name")),
        track_album_release_date=_coerce_str(row.get("track_album_release_date")),
        track_popularity=_coerce_int(row.get("track_popularity")),
        popularity_tier=_coerce_str(row.get("popularity_tier")) or "unknown",
        playlist_genres=_coerce_list(row.get("playlist_genres")),
        playlist_subgenres=_coerce_list(row.get("playlist_subgenres")),
        playlist_names=_coerce_list(row.get("playlist_names")),
        danceability=_coerce_float(row.get("danceability")),
        energy=_coerce_float(row.get("energy")),
        key=_coerce_int(row.get("key")),
        loudness=_coerce_float(row.get("loudness")),
        mode=_coerce_int(row.get("mode")),
        speechiness=_coerce_float(row.get("speechiness")),
        acousticness=_coerce_float(row.get("acousticness")),
        instrumentalness=_coerce_float(row.get("instrumentalness")),
        liveness=_coerce_float(row.get("liveness")),
        valence=_coerce_float(row.get("valence")),
        tempo=_coerce_float(row.get("tempo")),
        duration_ms=_coerce_int(row.get("duration_ms")),
        time_signature=_coerce_int(row.get("time_signature")),
        lyrics_clean=clean_lyrics(_coerce_str(row.get("lyrics"))),
        language=_coerce_str(row.get("language")),
        genius_tag=_coerce_str(row.get("genius_tag")),
    )
