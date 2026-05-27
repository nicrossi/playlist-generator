from playlist_rag.schemas import QueryIntent, RetrievedTrack


def rank_candidates(
    candidates: list[RetrievedTrack],
    intent: QueryIntent,
) -> list[RetrievedTrack]:
    """Score and sort candidates; mutates final_score on each track."""
    mood_set = set(intent.moods)
    energy_set = set(intent.energy_levels)

    for track in candidates:
        score = track.vector_score

        if mood_set and track.mood in mood_set:
            score += 0.12
        if energy_set and track.energy_qualitative in energy_set:
            score += 0.08

        if intent.prefer_obscure and track.popularity_tier == "obscure":
            score += 0.05
        if intent.prefer_popular and track.popularity_tier == "popular":
            score += 0.05

        if intent.min_instrumentalness is not None and track.instrumentalness is not None:
            if track.instrumentalness >= intent.min_instrumentalness:
                score += 0.08

        if intent.min_tempo is not None and track.tempo is not None:
            if track.tempo < intent.min_tempo:
                score -= 0.15
        if intent.max_tempo is not None and track.tempo is not None:
            if track.tempo > intent.max_tempo:
                score -= 0.15

        track.final_score = score

    return sorted(candidates, key=lambda t: t.final_score, reverse=True)
