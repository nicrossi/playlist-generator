from playlist_rag.schemas import PlaylistTrack, RetrievedTrack

_DESC_MAX = 400


def format_track_context(track: RetrievedTrack | PlaylistTrack) -> str:
    parts = [
        f"spotify_track_id={track.spotify_track_id}",
        f"title={track.track_name}",
        f"artist={track.track_artist}",
    ]
    if track.mood:
        parts.append(f"mood={track.mood}")
    if track.energy_qualitative:
        parts.append(f"energy={track.energy_qualitative}")
    if track.inferred_subgenre:
        parts.append(f"subgenre={track.inferred_subgenre}")
    desc = (track.description or "").strip()
    if len(desc) > _DESC_MAX:
        desc = desc[:_DESC_MAX] + "…"
    if desc:
        parts.append(f"description={desc}")
    return " | ".join(parts)


def format_context_block(tracks: list[RetrievedTrack | PlaylistTrack]) -> str:
    if not tracks:
        return "(no tracks)"
    return "\n".join(
        f"[{i + 1}] {format_track_context(t)}" for i, t in enumerate(tracks)
    )


def format_playlist_answer(
    query: str,
    tracks: list[PlaylistTrack],
    explanation: str,
    total_duration_minutes: float,
) -> str:
    lines = [
        f"User query: {query}",
        f"Playlist duration: ~{total_duration_minutes:.0f} min",
        f"Tracks ({len(tracks)}):",
    ]
    for t in tracks:
        lines.append(f"  {t.position}. {t.track_name} — {t.track_artist}")
    lines.append(f"Explanation: {explanation}")
    return "\n".join(lines)
