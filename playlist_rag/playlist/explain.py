from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from playlist_rag.config import settings
from playlist_rag.schemas import PlaylistTrack, QueryIntent

_EXPLAIN_SYSTEM = """You explain a generated music playlist to the user in 2-4 clear sentences.
Mention the overall vibe, how the selection matches their request, and one note about flow or diversity.
Do not list every song. Write in the same language the user used for their query."""

_EXPLAIN_USER = """User request: {query}

Parsed intent summary:
- semantic focus: {semantic_query}
- target duration: {duration} minutes
- moods: {moods}
- excluded artists: {exclude_artists}
- excluded genres: {exclude_genres}

Playlist ({track_count} tracks, ~{duration_actual:.0f} min):
{track_list}

Write a concise explanation."""

_llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.4,
    timeout=settings.llm_timeout_seconds,
    max_retries=2,
    api_key=settings.openai_api_key or None,
)

_prompt = ChatPromptTemplate.from_messages(
    [("system", _EXPLAIN_SYSTEM), ("user", _EXPLAIN_USER)]
)


def explain_playlist(
    user_query: str,
    intent: QueryIntent,
    tracks: list[PlaylistTrack],
    total_duration_minutes: float,
    use_llm: bool = True,
) -> str:
    if not tracks:
        return "No tracks matched your request. Try broadening filters or rephrasing."

    if not use_llm or not settings.openai_api_key:
        moods = ", ".join(intent.moods) if intent.moods else "varied"
        names = ", ".join(f"{t.track_name} ({t.track_artist})" for t in tracks[:5])
        suffix = "…" if len(tracks) > 5 else ""
        return (
            f"Playlist of {len(tracks)} tracks (~{total_duration_minutes:.0f} min) "
            f"matching '{intent.semantic_query}' with mood focus: {moods}. "
            f"Includes: {names}{suffix}"
        )

    track_list = "\n".join(
        f"{t.position}. {t.track_name} — {t.track_artist}" for t in tracks[:15]
    )
    chain = _prompt | _llm
    msg = chain.invoke(
        {
            "query": user_query,
            "semantic_query": intent.semantic_query,
            "duration": intent.target_duration_minutes
            or settings.default_duration_minutes,
            "moods": ", ".join(intent.moods) or "(none)",
            "exclude_artists": ", ".join(intent.exclude_artists) or "(none)",
            "exclude_genres": ", ".join(intent.exclude_genres) or "(none)",
            "track_count": len(tracks),
            "duration_actual": total_duration_minutes,
            "track_list": track_list,
        }
    )
    return str(msg.content).strip()
