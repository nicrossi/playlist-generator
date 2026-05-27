# Generation Pipeline (`playlist_rag`)

The online half of the RAG system. Takes a natural-language playlist request,
parses it into structured intent, retrieves candidate tracks from the indexed
catalog (Postgres + pgvector), ranks and sequences them into a coherent
playlist, and returns a textual explanation.

**Prerequisite:** a fully indexed catalog. See
[`indexing_pipeline.md`](indexing_pipeline.md) — especially the HNSW migration
(`alembic upgrade head`) after bulk indexing.

```
user query (NL)
        │
        ▼
[Stage 1: parse]       ── ChatOpenAI + PydanticOutputParser
        │  QueryIntent
        ▼
[Stage 2: embed]       ── OpenAIEmbeddings (same model as indexing)
        │  query vector (1536-D)
        ▼
[Stage 3: retrieve]    ── hybrid_search: cosine + SQL filters
        │  list[RetrievedTrack]  (top-k, default 80)
        ▼
[Stage 4: rank]        ── heuristic boosts on vector_score
        ▼
[Stage 5: sequence]    ── greedy: duration + transitions + artist spacing
        │  list[PlaylistTrack]
        ▼
[Stage 6: explain]     ── LLM or template summary
        │
        ▼
   PlaylistResult
```

Each request makes **two or three** OpenAI calls: parse (1), query embed (1),
explain (0–1). Retrieval and ranking are local SQL + Python.

## Quickstart

```bash
# Prerequisites (once)
docker compose up -d
cp .env.example .env          # set OPENAI_API_KEY
alembic upgrade head          # tables + HNSW index
# ... index catalog (see indexing_pipeline.md)

# Generate
python -m playlist_rag.cli.generate "calm instrumental music for studying 30 minutes"
python -m playlist_rag.cli.generate "música tranquila sin reggaetón 15 minutos" --json
python -m playlist_rag.cli.generate "upbeat rock for a workout" --no-llm-explain --top-k 50
```

Exit codes: `0` = playlist produced, `1` = error, `2` = zero tracks matched.

## Module layout

```
playlist_rag/
├── schemas.py              QueryIntent, RetrievedTrack, PlaylistTrack, PlaylistResult
├── query/
│   └── parse.py            NL → QueryIntent
├── retrieval/
│   ├── filters.py          QueryIntent → SQL WHERE fragment
│   └── search.py           hybrid_search()
├── playlist/
│   ├── rank.py             rank_candidates()
│   ├── sequence.py         build_playlist()
│   ├── explain.py          explain_playlist()
│   └── pipeline.py         generate_playlist() orchestrator
└── cli/
    └── generate.py         `python -m playlist_rag.cli.generate`
```

Reuses from indexing: `indexing/embed.py` (query embedding), `db.py` (sessions),
`config.py` (settings).

## Stages

### Stage 1 — Parse (`query/parse.py`)

`parse_query(user_query: str) → QueryIntent`

LangChain `ChatOpenAI` + `PydanticOutputParser` at temperature `0.1`. The LLM
rewrites the user's request into:

- **`semantic_query`** — English, 1–3 sentences, used for embedding search
  (must align with how track descriptions were written during indexing).
- **Structured filters** — moods, energy, tempo/energy bounds, exclusions, etc.

**Retry policy** (`@retry` via tenacity): one retry on `OutputParserException` /
`ValidationError`; API errors propagate.

**Parser rules (high level):**

| User says | Field |
|---|---|
| "2 hours", "45 min" | `target_duration_minutes` |
| "sin Kanye", "no reggaeton" | `exclude_artists` / `exclude_genres` |
| "instrumental", "sin voces" | `min_instrumentalness=0.5` (not `include_genres`) |
| "solo rock" | `include_genres=["rock"]` (closed playlist-genre list only) |
| "en español" (explicit) | `languages=["es"]` |
| Query written in Spanish | **does not** set `languages` |

Closed sets match indexing: `Mood` and `EnergyQualitative` in `schemas.py`.

### Stage 2 — Embed query

`embed_text(intent.semantic_query)` from `indexing/embed.py`.

Same model and dimension as track embeddings (`text-embedding-3-small`, 1536-D).
Mismatch with indexed vectors would break retrieval quality.

### Stage 3 — Hybrid retrieval (`retrieval/search.py`)

`hybrid_search(session, intent, query_vector, top_k=80)`

Single SQL query:

1. Join `tracks` ↔ `track_embeddings`.
2. Apply dynamic `WHERE` from `build_filter_clauses(intent)` (`filters.py`).
3. Order by cosine distance (`<=>` operator; HNSW index on `embedding`).
4. Return `vector_score = 1 - distance` per row.

**Filter mapping:**

| `QueryIntent` field | SQL |
|---|---|
| `moods` | `t.mood = ANY(:moods)` |
| `energy_levels` | `t.energy_qualitative = ANY(:energy_levels)` |
| `languages` | `LOWER(t.language) = ANY(:languages)` |
| `min/max_tempo`, `min/max_energy` | range on `t.tempo` / `t.energy` |
| `exclude_artists` | `track_artist NOT ILIKE %name%` per artist |
| `exclude_genres` | NOT overlap on `playlist_genres`, `playlist_subgenres`, or `inferred_subgenre ILIKE` |
| `include_genres` | `playlist_genres && ARRAY[...]` |
| `min_instrumentalness` | `t.instrumentalness >= threshold` |
| `prefer_obscure` / `prefer_popular` | `popularity_tier = 'obscure' \| 'popular'` |

**Zero-result fallback** (`playlist/pipeline.py`): if strict filters return no
rows, retry retrieval with moods, energy, languages, include_genres, and
`min_instrumentalness` cleared. **Exclusions are never relaxed.**

### Stage 4 — Rank (`playlist/rank.py`)

`rank_candidates(candidates, intent) → sorted list`

Starts from `vector_score`, then additive adjustments:

| Signal | Δ score |
|---|---|
| mood match | +0.12 |
| energy_qualitative match | +0.08 |
| instrumentalness ≥ threshold | +0.08 |
| popularity tier preference | +0.05 |
| tempo out of range | −0.15 |

Writes `final_score` on each `RetrievedTrack` and sorts descending.

### Stage 5 — Sequence (`playlist/sequence.py`)

`build_playlist(ranked, intent) → list[PlaylistTrack]`

Greedy construction until duration target or `max_playlist_tracks` (50):

- **Duration target:** `target_duration_minutes` from intent, else
  `default_duration_minutes` (60). Missing `duration_ms` → 180 s assumed.
- **Per-step score:** `0.65 × final_score + 0.35 × transition_score`.
- **Transition:** penalizes large tempo and energy jumps vs. previous track.
- **Artist spacing:** skips artists seen in the last `artist_spacing` (2) picks;
  relaxes spacing if the pool would stall.

Each output track gets `position` and a short `reason` string (template).

### Stage 6 — Explain (`playlist/explain.py`)

`explain_playlist(query, intent, tracks, duration, use_llm=True)`

- **LLM mode (default):** 2–4 sentences in the **user's query language**,
  summarizing vibe and fit (does not list every song).
- **Template mode** (`--no-llm-explain` or missing API key): one paragraph with
  track count, duration, semantic focus, and first five titles.
- **Empty playlist:** fixed message suggesting broader phrasing.

## Schemas

### `QueryIntent`

Structured search + filter spec produced by Stage 1. See `schemas.py` for the
full field list.

### `RetrievedTrack`

Candidate from DB with `vector_score` and optional `final_score` after ranking.

### `PlaylistTrack`

`RetrievedTrack` + `position` + `reason`.

### `PlaylistResult`

```python
class PlaylistResult(BaseModel):
    query: str
    intent: QueryIntent
    tracks: list[PlaylistTrack]
    total_duration_minutes: float
    explanation: str
```

Use `--json` on the CLI for `model_dump_json()` output.

## CLI

```bash
python -m playlist_rag.cli.generate QUERY [OPTIONS]
```

| Flag | Purpose |
|---|---|
| `QUERY` | Natural-language request (required) |
| `--json` | Full `PlaylistResult` as JSON |
| `--no-llm-explain` | Skip explanation LLM call |
| `--top-k N` | Retrieval pool size (default `retrieval_top_k=80`) |
| `-v` | DEBUG logging |

**Programmatic use:**

```python
from playlist_rag.playlist.pipeline import generate_playlist

result = generate_playlist(
    "melancholic late-night drives",
    use_llm_explain=True,
    top_k=80,
)
```

## Configuration

Generation-specific settings in `playlist_rag/config.py` (via `.env` or defaults):

| Field | Default | Notes |
|---|---|---|
| `retrieval_top_k` | `80` | Candidates before rank/sequence |
| `default_duration_minutes` | `60.0` | When user omits length |
| `max_playlist_tracks` | `50` | Hard cap on output size |
| `artist_spacing` | `2` | Min gap between same artist |

Shared with indexing: `openai_api_key`, `llm_model`, `embedding_model`,
`embedding_dim`, `database_url`.

## Error handling

| Failure | Behavior |
|---|---|
| Empty user query | `ValueError` from `parse_query` |
| Malformed LLM parse JSON | Tenacity retry once, then raise |
| Zero retrieval results (after fallback) | Empty playlist, exit code `2` |
| Missing `OPENAI_API_KEY` | Fails at parse/embed; explain falls back to template if only explain needs key |
| Postgres down | SQLAlchemy error; `pool_pre_ping=True` on reconnect |

## Known limitations (MVP)

- **Parser quality** drives filter accuracy; wrong `include_genres` or inferred
  `languages` can over-filter (mitigated by zero-result fallback for non-exclusion
  filters).
- **`include_genres`** only matches Kaggle playlist genre tags (e.g. `rock`,
  `latin`), not free-text labels like "instrumental".
- **`exclude_genres`** depends on metadata (`playlist_*`, `inferred_subgenre`);
  tracks without reggaeton tags may still slip through.
- **Sequencing** is greedy, not globally optimal; no explicit diversity beyond
  artist spacing.
- **No user profile**, cold-start survey, on-demand track ingestion, or UI.

## Relationship to other docs

| Doc | Role |
|---|---|
| [`indexing_pipeline.md`](indexing_pipeline.md) | Builds the catalog this pipeline reads |
| [`dataset_unification.md`](dataset_unification.md) | Upstream parquet schema |
| [`decisions.md`](decisions.md) | Design rationale (pgvector, structured LLM output, etc.) |
