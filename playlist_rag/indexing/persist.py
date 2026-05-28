from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from playlist_rag.config import settings
from playlist_rag.db import Track, TrackEmbedding, TrackIndexStatus, TrackTheme
from playlist_rag.schemas import NormalizedTrack, TrackSemantics


def _track_values(norm: NormalizedTrack, sem: TrackSemantics) -> dict:
    return {
        "spotify_track_id": norm.spotify_track_id,
        "track_name": norm.track_name,
        "track_artist": norm.track_artist,
        "track_album_name": norm.track_album_name,
        "track_album_release_date": norm.track_album_release_date,
        "danceability": norm.danceability,
        "energy": norm.energy,
        "key": norm.key,
        "loudness": norm.loudness,
        "mode": norm.mode,
        "speechiness": norm.speechiness,
        "acousticness": norm.acousticness,
        "instrumentalness": norm.instrumentalness,
        "liveness": norm.liveness,
        "valence": norm.valence,
        "tempo": norm.tempo,
        "duration_ms": norm.duration_ms,
        "time_signature": norm.time_signature,
        "track_popularity": norm.track_popularity,
        "popularity_tier": norm.popularity_tier,
        "playlist_genres": norm.playlist_genres,
        "playlist_subgenres": norm.playlist_subgenres,
        "playlist_names": norm.playlist_names,
        "lyrics": norm.lyrics_clean,
        "language": norm.language,
        "genius_tag": norm.genius_tag,
        "description": sem.description,
        "mood": sem.mood,
        "inferred_subgenre": sem.inferred_subgenre,
        "energy_qualitative": sem.energy_qualitative,
        "indexer_version": settings.indexer_version,
    }


def upsert_track(
    session: Session,
    norm: NormalizedTrack,
    sem: TrackSemantics,
    embedding: list[float],
) -> int:
    values = _track_values(norm, sem)
    update_cols = {k: v for k, v in values.items() if k != "spotify_track_id"}
    update_cols["indexed_at"] = datetime.now(timezone.utc)

    stmt = (
        pg_insert(Track)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["spotify_track_id"], set_=update_cols
        )
        .returning(Track.id)
    )
    track_id = session.execute(stmt).scalar_one()

    emb_stmt = (
        pg_insert(TrackEmbedding)
        .values(
            track_id=track_id,
            embedding=embedding,
            model_name=settings.embedding_model,
        )
        .on_conflict_do_update(
            index_elements=["track_id"],
            set_={
                "embedding": embedding,
                "model_name": settings.embedding_model,
                "embedded_at": datetime.now(timezone.utc),
            },
        )
    )
    session.execute(emb_stmt)

    session.execute(delete(TrackTheme).where(TrackTheme.track_id == track_id))
    unique_themes = {t.strip() for t in sem.themes if t and t.strip()}
    if unique_themes:
        session.execute(
            pg_insert(TrackTheme),
            [{"track_id": track_id, "theme": t} for t in unique_themes],
        )

    return track_id


def get_status(
    session: Session, spotify_track_id: str
) -> Optional[TrackIndexStatus]:
    return session.get(TrackIndexStatus, spotify_track_id)


def mark_status(
    session: Session,
    spotify_track_id: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc)
    insert_values = {
        "spotify_track_id": spotify_track_id,
        "status": status,
        "last_error": error,
        "last_attempt_at": now,
        "attempt_count": 1 if status == "failed" else 0,
    }
    update_values: dict = {
        "status": status,
        "last_error": error,
        "last_attempt_at": now,
    }
    if status == "failed":
        update_values["attempt_count"] = TrackIndexStatus.attempt_count + 1

    stmt = (
        pg_insert(TrackIndexStatus)
        .values(**insert_values)
        .on_conflict_do_update(
            index_elements=["spotify_track_id"], set_=update_values
        )
    )
    session.execute(stmt)


def get_completed_ids(session: Session) -> set[str]:
    rows = session.execute(
        select(TrackIndexStatus.spotify_track_id).where(
            TrackIndexStatus.status == "complete"
        )
    ).all()
    return {r[0] for r in rows}


def delete_index_status(session: Session, spotify_track_ids: list[str]) -> int:
    if not spotify_track_ids:
        return 0
    result = session.execute(
        delete(TrackIndexStatus).where(
            TrackIndexStatus.spotify_track_id.in_(spotify_track_ids)
        )
    )
    return result.rowcount or 0
