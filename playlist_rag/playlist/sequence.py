from playlist_rag.config import settings
from playlist_rag.schemas import PlaylistTrack, QueryIntent, RetrievedTrack

_DEFAULT_DURATION_MS = 180_000


def _transition_score(prev: RetrievedTrack, nxt: RetrievedTrack) -> float:
    """Higher is smoother transition (0..1)."""
    if prev.tempo is None or nxt.tempo is None:
        tempo_penalty = 0.0
    else:
        tempo_penalty = min(abs(prev.tempo - nxt.tempo) / 60.0, 1.0)

    if prev.energy is None or nxt.energy is None:
        energy_penalty = 0.0
    else:
        energy_penalty = min(abs(prev.energy - nxt.energy) / 0.5, 1.0)

    return 1.0 - 0.5 * tempo_penalty - 0.5 * energy_penalty


def _track_reason(track: RetrievedTrack, intent: QueryIntent) -> str:
    parts = [f"semantic match ({track.vector_score:.2f})"]
    if intent.moods and track.mood in intent.moods:
        parts.append(f"mood={track.mood}")
    if intent.energy_levels and track.energy_qualitative in intent.energy_levels:
        parts.append(f"energy={track.energy_qualitative}")
    if track.inferred_subgenre:
        parts.append(track.inferred_subgenre)
    return "; ".join(parts)


def build_playlist(
    ranked: list[RetrievedTrack],
    intent: QueryIntent,
) -> list[PlaylistTrack]:
    """Greedy sequencer: duration target + transition smoothness + artist spacing."""
    target_ms = int(
        (intent.target_duration_minutes or settings.default_duration_minutes)
        * 60
        * 1000
    )
    max_tracks = settings.max_playlist_tracks
    spacing = settings.artist_spacing

    pool = list(ranked)
    ordered: list[RetrievedTrack] = []
    recent_artists: list[str] = []
    total_ms = 0

    while pool and total_ms < target_ms and len(ordered) < max_tracks:
        best: RetrievedTrack | None = None
        best_combined = -1.0
        prev = ordered[-1] if ordered else None

        for candidate in pool:
            if candidate.track_artist in recent_artists[-spacing:]:
                continue

            relevance = candidate.final_score
            transition = _transition_score(prev, candidate) if prev else 1.0
            combined = 0.65 * relevance + 0.35 * transition

            if combined > best_combined:
                best = candidate
                best_combined = combined

        if best is None:
            # relax artist spacing
            for candidate in pool:
                relevance = candidate.final_score
                transition = _transition_score(prev, candidate) if prev else 1.0
                combined = 0.65 * relevance + 0.35 * transition
                if combined > best_combined:
                    best = candidate
                    best_combined = combined

        if best is None:
            break

        pool.remove(best)
        ordered.append(best)
        recent_artists.append(best.track_artist)
        total_ms += best.duration_ms or _DEFAULT_DURATION_MS

    return [
        PlaylistTrack(
            **best.model_dump(),
            position=i + 1,
            reason=_track_reason(best, intent),
        )
        for i, best in enumerate(ordered)
    ]
