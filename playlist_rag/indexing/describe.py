from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from playlist_rag.config import settings
from playlist_rag.schemas import NormalizedTrack, TrackSemantics

_SYSTEM_PROMPT = """You are an expert music analyst writing concise, evocative semantic descriptions of tracks for a recommendation system.

Your descriptions inform downstream retrieval and playlist sequencing, so they must be specific and information-dense. Avoid generic phrasing like "this song explores feelings" or "an emotional track." Name the texture, the instrumentation, the lyrical posture.

When audio features are provided, interpret them concretely:
- high energy + high valence: upbeat, party, celebration
- high energy + low valence: angry, intense, aggressive
- low energy + high valence: calm, content, peaceful
- low energy + low valence: sad, melancholic, introspective
- high acousticness: organic, intimate, stripped-down
- high instrumentalness: ambient, focus, no vocals to follow
- high speechiness: spoken-word leaning, rap, or talky delivery
- high danceability with high tempo: club, propulsive, dance
- low danceability with low tempo: slow-burn, ballad, contemplative

Themes must be short lowercase noun phrases. Mood and energy_qualitative must come from the closed enums provided in the format instructions. inferred_subgenre is free text and may extend beyond the playlist tags when warranted.

{format_instructions}
"""

_USER_FULL = """Track metadata:
- Title: {track_name}
- Artist: {track_artist}
- Album: {track_album_name}
- Release date: {release_date}
- Popularity tier: {popularity_tier}
- Playlist genres: {playlist_genres}
- Playlist subgenres: {playlist_subgenres}
- Language: {language}
- Genius tag: {genius_tag}

Audio features:
- danceability: {danceability}
- energy: {energy}
- valence: {valence}
- tempo: {tempo} BPM
- acousticness: {acousticness}
- instrumentalness: {instrumentalness}
- speechiness: {speechiness}
- liveness: {liveness}
- loudness: {loudness} dB
- key: {key}, mode: {mode}, time_signature: {time_signature}

Lyrics excerpt:
{lyrics_excerpt}

Produce the structured output. Write the description in English."""

_USER_DEGRADED = """No lyrics are available for this track. Base your description on the title, artist, audio features, and genre tags. Be especially careful with themes — infer them conservatively when lyrics aren't available.

Track metadata:
- Title: {track_name}
- Artist: {track_artist}
- Album: {track_album_name}
- Release date: {release_date}
- Popularity tier: {popularity_tier}
- Playlist genres: {playlist_genres}
- Playlist subgenres: {playlist_subgenres}

Audio features:
- danceability: {danceability}
- energy: {energy}
- valence: {valence}
- tempo: {tempo} BPM
- acousticness: {acousticness}
- instrumentalness: {instrumentalness}
- speechiness: {speechiness}
- liveness: {liveness}
- loudness: {loudness} dB
- key: {key}, mode: {mode}, time_signature: {time_signature}

Produce the structured output. Write the description in English."""


_parser = PydanticOutputParser(pydantic_object=TrackSemantics)
_format_instructions = _parser.get_format_instructions()

_llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    timeout=settings.llm_timeout_seconds,
    max_retries=0,
    api_key=settings.openai_api_key or None,
)

_prompt_with_lyrics = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("user", _USER_FULL)]
).partial(format_instructions=_format_instructions)

_prompt_no_lyrics = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("user", _USER_DEGRADED)]
).partial(format_instructions=_format_instructions)

_chain_with_lyrics = _prompt_with_lyrics | _llm | _parser
_chain_no_lyrics = _prompt_no_lyrics | _llm | _parser


def _fmt_float(v: Any) -> str:
    if v is None:
        return "Unknown"
    return f"{float(v):.3f}"


def _fmt_int(v: Any) -> str:
    if v is None:
        return "Unknown"
    return str(int(v))


def _fmt_str(v: Any) -> str:
    return v if v else "Unknown"


def _fmt_list(v: list[str]) -> str:
    return ", ".join(v) if v else "(none)"


def _build_inputs(track: NormalizedTrack) -> dict:
    inputs = {
        "track_name": track.track_name,
        "track_artist": track.track_artist,
        "track_album_name": _fmt_str(track.track_album_name),
        "release_date": _fmt_str(track.track_album_release_date),
        "popularity_tier": track.popularity_tier,
        "playlist_genres": _fmt_list(track.playlist_genres),
        "playlist_subgenres": _fmt_list(track.playlist_subgenres),
        "danceability": _fmt_float(track.danceability),
        "energy": _fmt_float(track.energy),
        "valence": _fmt_float(track.valence),
        "tempo": _fmt_float(track.tempo),
        "acousticness": _fmt_float(track.acousticness),
        "instrumentalness": _fmt_float(track.instrumentalness),
        "speechiness": _fmt_float(track.speechiness),
        "liveness": _fmt_float(track.liveness),
        "loudness": _fmt_float(track.loudness),
        "key": _fmt_int(track.key),
        "mode": _fmt_int(track.mode),
        "time_signature": _fmt_int(track.time_signature),
    }
    if track.has_lyrics:
        excerpt = (track.lyrics_clean or "")[: settings.lyrics_excerpt_chars]
        inputs["language"] = _fmt_str(track.language)
        inputs["genius_tag"] = _fmt_str(track.genius_tag)
        inputs["lyrics_excerpt"] = excerpt
    return inputs


@retry(
    retry=retry_if_exception_type((OutputParserException, ValidationError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def describe(track: NormalizedTrack) -> TrackSemantics:
    chain = _chain_with_lyrics if track.has_lyrics else _chain_no_lyrics
    return chain.invoke(_build_inputs(track))
