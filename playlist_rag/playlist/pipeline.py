from playlist_rag.db import get_session
from playlist_rag.indexing.embed import embed_text
from playlist_rag.playlist.explain import explain_playlist
from playlist_rag.playlist.rank import rank_candidates
from playlist_rag.playlist.sequence import build_playlist
from playlist_rag.query.parse import parse_query
from playlist_rag.retrieval.search import hybrid_search
from playlist_rag.schemas import PlaylistResult

_DEFAULT_DURATION_MS = 180_000


def _total_duration_minutes(tracks: list) -> float:
    total_ms = sum(t.duration_ms or _DEFAULT_DURATION_MS for t in tracks)
    return total_ms / 60_000.0


def generate_playlist(
    user_query: str,
    *,
    use_llm_explain: bool = True,
    top_k: int | None = None,
) -> PlaylistResult:
    intent = parse_query(user_query)
    query_vector = embed_text(intent.semantic_query)

    with get_session() as session:
        candidates = hybrid_search(session, intent, query_vector, top_k=top_k)
        if not candidates and (
            intent.moods
            or intent.energy_levels
            or intent.languages
            or intent.include_genres
            or intent.min_instrumentalness is not None
        ):
            relaxed = intent.model_copy(
                update={
                    "moods": [],
                    "energy_levels": [],
                    "languages": [],
                    "include_genres": [],
                    "min_instrumentalness": None,
                }
            )
            candidates = hybrid_search(
                session, relaxed, query_vector, top_k=top_k
            )

    ranked = rank_candidates(candidates, intent)
    playlist_tracks = build_playlist(ranked, intent)
    duration = _total_duration_minutes(playlist_tracks)
    explanation = explain_playlist(
        user_query, intent, playlist_tracks, duration, use_llm=use_llm_explain
    )

    return PlaylistResult(
        query=user_query,
        intent=intent,
        tracks=playlist_tracks,
        total_duration_minutes=duration,
        explanation=explanation,
    )
