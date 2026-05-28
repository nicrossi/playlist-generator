"""Streamlit UI for the playlist generator.

Thin presentation layer over `generate_playlist_with_trace()`: takes a
natural-language query, runs the full pipeline, and shows the explanation, the
sequenced playlist, what the model parsed from the query, and the retrieval
trace (candidate pool).

Run: streamlit run playlist_rag/ui/app.py
"""

import html
import sys
from pathlib import Path

# `streamlit run` only puts this file's dir on sys.path, not the repo root,
# so the `playlist_rag` package import below fails without this bootstrap.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from playlist_rag.config import settings  # noqa: E402
from playlist_rag.playlist.pipeline import generate_playlist_with_trace  # noqa: E402
from playlist_rag.schemas import (  # noqa: E402
    GenerationRun,
    GenerationTrace,
    PlaylistResult,
    QueryIntent,
)

SPOTIFY_TRACK_URL = "https://open.spotify.com/track/{}"
PLACEHOLDER = "música tranquila para estudiar 2 horas sin reggaetón"

# Spotify-like palette. Theme (.streamlit/config.toml) sets base dark colors;
# this CSS adds the green accents, pill button, and card surfaces.
GREEN = "#1DB954"
GREEN_HOVER = "#1ED760"
SURFACE = "#181818"
SUBDUED = "#B3B3B3"

_CSS = f"""
<style>
.stApp h1 {{ color: #FFFFFF; font-weight: 800; letter-spacing: -0.5px; }}
.stApp h2, .stApp h3 {{ color: #FFFFFF; font-weight: 700; }}
div[data-testid="stCaptionContainer"] {{ color: {SUBDUED}; }}

/* Pill-shaped Spotify green primary button */
button[kind="primary"] {{
    background: {GREEN};
    border: none;
    border-radius: 500px;
    color: #000000;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 0.55rem 2rem;
    transition: transform 0.1s ease, background 0.1s ease;
}}
button[kind="primary"]:hover {{
    background: {GREEN_HOVER};
    transform: scale(1.04);
    color: #000000;
}}

div[data-testid="stExpander"] {{ border-radius: 8px; }}
.explain {{
    background: {SURFACE};
    border-left: 4px solid {GREEN};
    padding: 0.9rem 1.2rem;
    border-radius: 8px;
    line-height: 1.55;
    color: #FFFFFF;
}}
</style>
"""


@st.cache_data(show_spinner=False)
def _generate(query: str, use_llm_explain: bool, top_k: int) -> GenerationRun:
    return generate_playlist_with_trace(
        query, use_llm_explain=use_llm_explain, top_k=top_k
    )


def _format_duration(duration_ms: int | None) -> str:
    seconds = round((duration_ms or 0) / 1000)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _tracks_dataframe(result: PlaylistResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "#": t.position,
                "Title": t.track_name,
                "Artist": t.track_artist,
                "Mood": t.mood,
                "Energy": t.energy_qualitative,
                "Tempo": round(t.tempo) if t.tempo is not None else None,
                "Duration": _format_duration(t.duration_ms),
                "Why": t.reason,
                "Spotify": SPOTIFY_TRACK_URL.format(t.spotify_track_id),
            }
            for t in result.tracks
        ]
    )


def _candidates_dataframe(trace: GenerationTrace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Title": c.track_name,
                "Artist": c.track_artist,
                "Mood": c.mood,
                "Subgenre": c.inferred_subgenre,
                "Match": round(c.vector_score, 3),
            }
            for c in trace.retrieved_candidates
        ]
    )


def _intent_summary(intent: QueryIntent) -> dict:
    """Drop unset fields (None, empty list, False) for a clean display."""
    return {
        k: v
        for k, v in intent.model_dump().items()
        if v not in (None, [], False, "")
    }


def main() -> None:
    st.set_page_config(page_title="Playlist Generator", page_icon="🎵", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("🎵 Playlist Generator")
    st.caption("Natural-language query → sequenced playlist with explanations.")

    if not settings.openai_api_key:
        st.warning(
            "OPENAI_API_KEY is not set. Query parsing and retrieval need it — "
            "set it in .env before generating."
        )

    with st.form("query"):
        query = st.text_area("What do you want to listen to?", placeholder=PLACEHOLDER)
        left, right = st.columns(2)
        top_k = left.slider(
            "Retrieval pool size",
            min_value=20,
            max_value=200,
            value=settings.retrieval_top_k,
            step=10,
            help="Candidates fetched before ranking and sequencing.",
        )
        use_llm = right.toggle(
            "LLM explanation",
            value=True,
            help="Off uses a template summary (no extra OpenAI call).",
        )
        submitted = st.form_submit_button("Generate", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Enter a query first.")
            st.stop()
        try:
            with st.spinner("Generating playlist…"):
                st.session_state.run = _generate(query.strip(), use_llm, top_k)
        except Exception as e:
            st.session_state.pop("run", None)
            st.error(f"Generation failed: {e}")
            st.stop()

    run: GenerationRun | None = st.session_state.get("run")
    if run is None:
        return
    result, trace = run.result, run.trace

    if not result.tracks:
        st.info("No tracks matched that query. Try loosening the constraints.")
        return

    st.subheader("Why this playlist")
    st.markdown(
        f'<div class="explain">{html.escape(result.explanation)}</div>',
        unsafe_allow_html=True,
    )

    summary = f"{len(result.tracks)} tracks · ~{result.total_duration_minutes:.0f} min"
    summary += f" · {len(trace.retrieved_candidates)} candidates retrieved"
    if trace.retrieval_relaxed:
        summary += " · filters relaxed to fill the pool"
    st.caption(summary)

    st.dataframe(
        _tracks_dataframe(result),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Why": st.column_config.TextColumn(width="large"),
            "Spotify": st.column_config.LinkColumn(display_text="open ↗"),
        },
    )

    with st.expander("What the model understood"):
        st.json(_intent_summary(result.intent))

    with st.expander(f"Retrieval pool ({len(trace.retrieved_candidates)} candidates)"):
        st.dataframe(
            _candidates_dataframe(trace),
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
