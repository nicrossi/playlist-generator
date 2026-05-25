import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm

from playlist_rag.db import get_session
from playlist_rag.indexing.describe import describe
from playlist_rag.indexing.embed import embed_text
from playlist_rag.indexing.normalize import normalize_row
from playlist_rag.indexing.persist import (
    get_completed_ids,
    mark_status,
    upsert_track,
)
from playlist_rag.schemas import NormalizedTrack

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    total: int = 0
    skipped_complete: int = 0
    succeeded: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def processed(self) -> int:
        return self.succeeded + self.failed

    def __str__(self) -> str:
        rate = (
            self.processed / self.elapsed_seconds if self.elapsed_seconds else 0
        )
        return (
            f"RunStats(total={self.total}, "
            f"skipped={self.skipped_complete}, "
            f"succeeded={self.succeeded}, "
            f"failed={self.failed}, "
            f"elapsed={self.elapsed_seconds:.1f}s, "
            f"rate={rate:.2f} tracks/s)"
        )


def _iter_parquet(
    parquet_path: Path, limit: Optional[int] = None
) -> Iterator[dict]:
    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)
    for _, row in df.iterrows():
        yield row.to_dict()


def _process_one(norm: NormalizedTrack) -> tuple[bool, Optional[str]]:
    try:
        semantics = describe(norm)
        embedding = embed_text(semantics.description)
        with get_session() as session:
            upsert_track(session, norm, semantics, embedding)
            mark_status(session, norm.spotify_track_id, "complete")
        return True, None
    except Exception as e:
        logger.exception(
            "Track failed: spotify_track_id=%s", norm.spotify_track_id
        )
        err = str(e)[:500]
        try:
            with get_session() as session:
                mark_status(
                    session, norm.spotify_track_id, "failed", error=err
                )
        except Exception:
            logger.exception(
                "Failed to record failure status for %s",
                norm.spotify_track_id,
            )
        return False, err


def run_pipeline(
    parquet_path: Path,
    limit: Optional[int] = None,
    retry_failed: bool = False,
) -> RunStats:
    stats = RunStats()
    start = perf_counter()

    with get_session() as session:
        completed = get_completed_ids(session)
    logger.info("Skip set built: %d tracks already complete", len(completed))

    iterator = _iter_parquet(parquet_path, limit=limit)
    for row in tqdm(iterator, desc="Indexing", unit="track"):
        stats.total += 1
        spotify_id = str(row["track_id"])

        if spotify_id in completed and not retry_failed:
            stats.skipped_complete += 1
            continue

        try:
            norm = normalize_row(row)
        except Exception as e:
            logger.exception("Normalize failed for %s", spotify_id)
            try:
                with get_session() as session:
                    mark_status(
                        session,
                        spotify_id,
                        "failed",
                        error=f"normalize: {e!s}"[:500],
                    )
            except Exception:
                logger.exception(
                    "Failed to record normalize-failure status for %s",
                    spotify_id,
                )
            stats.failed += 1
            continue

        ok, _ = _process_one(norm)
        if ok:
            stats.succeeded += 1
        else:
            stats.failed += 1

    stats.elapsed_seconds = perf_counter() - start
    return stats
