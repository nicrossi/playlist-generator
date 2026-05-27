"""Hybrid vector + structured filter search over indexed tracks."""

from sqlalchemy import text
from sqlalchemy.orm import Session

from playlist_rag.config import settings
from playlist_rag.retrieval.filters import build_filter_clauses
from playlist_rag.schemas import QueryIntent, RetrievedTrack


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"


def hybrid_search(
    session: Session,
    intent: QueryIntent,
    query_vector: list[float],
    top_k: int | None = None,
) -> list[RetrievedTrack]:
    """Return top-k tracks by cosine similarity with optional SQL filters."""
    k = top_k or settings.retrieval_top_k
    filter_sql, filter_params = build_filter_clauses(intent)
    vec_lit = _vector_literal(query_vector)

    sql = f"""
        SELECT
            t.id,
            t.spotify_track_id,
            t.track_name,
            t.track_artist,
            t.description,
            t.mood,
            t.energy_qualitative,
            t.inferred_subgenre,
            t.tempo,
            t.energy,
            t.valence,
            t.instrumentalness,
            t.duration_ms,
            t.popularity_tier,
            1 - (e.embedding <=> CAST(:query_vec AS vector)) AS vector_score
        FROM tracks t
        INNER JOIN track_embeddings e ON e.track_id = t.id
        WHERE t.description IS NOT NULL
        {filter_sql}
        ORDER BY e.embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
    """

    params = {"query_vec": vec_lit, "top_k": k, **filter_params}
    rows = session.execute(text(sql), params).mappings().all()

    return [
        RetrievedTrack(
            track_id=int(r["id"]),
            spotify_track_id=str(r["spotify_track_id"]),
            track_name=str(r["track_name"]),
            track_artist=str(r["track_artist"]),
            description=str(r["description"] or ""),
            mood=r["mood"],
            energy_qualitative=r["energy_qualitative"],
            inferred_subgenre=r["inferred_subgenre"],
            tempo=float(r["tempo"]) if r["tempo"] is not None else None,
            energy=float(r["energy"]) if r["energy"] is not None else None,
            valence=float(r["valence"]) if r["valence"] is not None else None,
            instrumentalness=(
                float(r["instrumentalness"])
                if r["instrumentalness"] is not None
                else None
            ),
            duration_ms=int(r["duration_ms"]) if r["duration_ms"] is not None else None,
            popularity_tier=r["popularity_tier"],
            vector_score=float(r["vector_score"] or 0.0),
        )
        for r in rows
    ]
