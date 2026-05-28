from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd
import requests
from tqdm import tqdm

from playlist_rag.db import get_session
from playlist_rag.ingest.lyrics_ovh import (
    BackfillStats,
    _CACHE_FOUND,
    _is_blank_lyrics,
    fetch_or_cache,
)
from playlist_rag.indexing.persist import delete_index_status

logger = logging.getLogger(__name__)


def backfill_parquet_lyrics(
    parquet_path: Path,
    *,
    out_path: Path | None = None,
    cache_dir: Path = Path("data/lyrics_cache"),
    api_base: str = "https://api.lyrics.ovh/v1",
    delay_seconds: float = 0.75,
    limit: int | None = None,
    dry_run: bool = False,
    use_cache: bool = True,
) -> BackfillStats:
    
    stats = BackfillStats()
    start = perf_counter()

    df = pd.read_parquet(parquet_path)
    missing_mask = df["lyrics"].apply(_is_blank_lyrics) if "lyrics" in df.columns else pd.Series(
        [True] * len(df)
    )
    missing_idx = df.index[missing_mask].tolist()
    stats.skipped_has_lyrics = int((~missing_mask).sum())
    if limit is not None:
        missing_idx = missing_idx[:limit]
    stats.candidates = len(missing_idx)

    if "lyrics_source" not in df.columns:
        df["lyrics_source"] = None

    http = requests.Session()
    row_iter = missing_idx
    if not dry_run:
        row_iter = tqdm(missing_idx, desc="Lyrics backfill", unit="track")

    for idx in row_iter:
        row = df.loc[idx]
        track_id = str(row["track_id"])
        artist = str(row.get("track_artist") or "")
        title = str(row.get("track_name") or "")

        if dry_run:
            continue

        result = fetch_or_cache(
            cache_dir,
            track_id,
            artist,
            title,
            api_base=api_base,
            delay_seconds=delay_seconds,
            session=http,
            use_cache=use_cache,
        )
        stats.fetched += 1
        if result.from_cache:
            stats.cache_hits += 1
        if result.status == _CACHE_FOUND and result.lyrics:
            df.at[idx, "lyrics"] = result.lyrics
            df.at[idx, "lyrics_source"] = "lyrics.ovh"
            stats.found += 1
            stats.backfilled_ids.append(track_id)
        elif result.status == "error":
            stats.errors += 1
        else:
            stats.not_found += 1

    if dry_run:
        logger.info(
            "Dry run: would attempt %d tracks without lyrics", len(missing_idx)
        )
        return stats

    destination = out_path or parquet_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)
    elapsed = perf_counter() - start
    logger.info(
        "Wrote %s (%d lyrics added, %.1fs)",
        destination,
        stats.found,
        elapsed,
    )
    return stats


def clear_index_status_for_reindex(spotify_track_ids: list[str]) -> int:
    if not spotify_track_ids:
        return 0
    batch_size = 500
    total = 0
    with get_session() as session:
        for i in range(0, len(spotify_track_ids), batch_size):
            batch = spotify_track_ids[i : i + batch_size]
            total += delete_index_status(session, batch)
    return total


def write_reindex_manifest(
    track_ids: list[str],
    manifest_path: Path,
    summary_path: Path | None,
    stats: BackfillStats,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(track_ids) + ("\n" if track_ids else ""),
        encoding="utf-8",
    )
    if summary_path:
        payload = asdict(stats)
        payload["reindex_track_ids_file"] = str(manifest_path)
        summary_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
