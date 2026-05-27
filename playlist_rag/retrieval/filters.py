"""Build SQL WHERE fragments and bind params from QueryIntent."""

from typing import Any

from playlist_rag.schemas import QueryIntent


def build_filter_clauses(intent: QueryIntent) -> tuple[str, dict[str, Any]]:
    """Return (sql_and_fragment, bind_params) to append after WHERE 1=1."""
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if intent.moods:
        clauses.append("t.mood = ANY(:moods)")
        params["moods"] = list(intent.moods)

    if intent.energy_levels:
        clauses.append("t.energy_qualitative = ANY(:energy_levels)")
        params["energy_levels"] = list(intent.energy_levels)

    if intent.languages:
        clauses.append("LOWER(t.language) = ANY(:languages)")
        params["languages"] = [lang.lower() for lang in intent.languages]

    if intent.min_tempo is not None:
        clauses.append("t.tempo >= :min_tempo")
        params["min_tempo"] = intent.min_tempo

    if intent.max_tempo is not None:
        clauses.append("t.tempo <= :max_tempo")
        params["max_tempo"] = intent.max_tempo

    if intent.min_energy is not None:
        clauses.append("t.energy >= :min_energy")
        params["min_energy"] = intent.min_energy

    if intent.max_energy is not None:
        clauses.append("t.energy <= :max_energy")
        params["max_energy"] = intent.max_energy

    for i, artist in enumerate(intent.exclude_artists):
        key = f"exclude_artist_{i}"
        clauses.append(f"LOWER(t.track_artist) NOT LIKE :{key}")
        params[key] = f"%{artist.lower()}%"

    if intent.exclude_genres:
        lowered = [g.lower() for g in intent.exclude_genres]
        clauses.append(
            """
            NOT (
                t.playlist_genres && CAST(:exclude_genres AS varchar[])
                OR t.playlist_subgenres && CAST(:exclude_genres AS varchar[])
                OR EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:exclude_genres AS varchar[])) AS eg(genre)
                    WHERE t.inferred_subgenre ILIKE '%' || eg.genre || '%'
                )
            )
            """
        )
        params["exclude_genres"] = lowered

    if intent.include_genres:
        clauses.append(
            "t.playlist_genres && CAST(:include_genres AS varchar[])"
        )
        params["include_genres"] = [g.lower() for g in intent.include_genres]

    if intent.min_instrumentalness is not None:
        clauses.append("t.instrumentalness >= :min_instrumentalness")
        params["min_instrumentalness"] = intent.min_instrumentalness

    if intent.prefer_obscure and not intent.prefer_popular:
        clauses.append("t.popularity_tier = 'obscure'")
    elif intent.prefer_popular and not intent.prefer_obscure:
        clauses.append("t.popularity_tier = 'popular'")

    if not clauses:
        return "", params

    return " AND " + " AND ".join(clauses), params
