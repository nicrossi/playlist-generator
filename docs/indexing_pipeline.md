# Indexing Pipeline (`playlist_rag`)

The indexing stage of the RAG system. Reads
`data/unified_tracks.parquet` (output of [`unify.py`](dataset_unification.md)),
generates a semantic description per track via an LLM, embeds it, and persists
everything into Postgres + pgvector. Downstream stages (retrieval, query
rewriting, playlist sequencing) read from the database it populates.

```
unified_tracks.parquet
        │
        ▼
[Stage 1: normalize]   ── pure functions, no I/O
        │  NormalizedTrack (Pydantic)
        ▼
[Stage 2: describe]    ── ChatOpenAI + PydanticOutputParser
        │  TrackSemantics (Pydantic)
        ▼
[Stage 3: embed]       ── OpenAIEmbeddings (text-embedding-3-small, 1536-D)
        │  list[float]
        ▼
[Stage 4: persist]     ── Postgres upsert (tracks + embeddings + themes)
        │
        ▼
   Postgres + pgvector
```

Per-track sessions, per-track error isolation, resumable, idempotent. One bad
row never rolls back hours of work.

## Quickstart

```bash
# 1. Install deps (in your venv)
pip install -r requirements.txt

# 2. Bring up Postgres + pgvector
docker compose up -d

# 3. Configure env
cp .env.example .env
# edit .env: set OPENAI_API_KEY

# 4. Create tables (NO HNSW yet — see "HNSW deferral" below)
alembic upgrade 0001_initial

# 5. Smoke-test on 20 tracks (~$0.01, ~1 minute)
python -m playlist_rag.cli.index --parquet data/unified_tracks.parquet --limit 20

# 6. Full run (~$1.50, ~1-2 hours)
python -m playlist_rag.cli.index --parquet data/unified_tracks.parquet

# 7. Build HNSW index on the populated table
alembic upgrade head
```

Re-running step 5 or 6 is a no-op: `track_index_status.status='complete'` rows
are skipped via an in-memory set built once at startup. Pass `--retry-failed`
to re-attempt rows previously marked failed. Mid-run Ctrl+C is safe — each
track commits in its own session.

## Module layout

```
playlist_rag/
├── config.py           pydantic-settings singleton (env-driven)
├── schemas.py          TrackSemantics + NormalizedTrack
├── db.py               SQLAlchemy models + get_session()
├── indexing/
│   ├── normalize.py    parquet row → NormalizedTrack
│   ├── describe.py     LLM + PydanticOutputParser, two prompt variants
│   ├── embed.py        OpenAIEmbeddings wrapper
│   ├── persist.py      upserts + status tracking
│   └── pipeline.py     orchestrator
└── cli/
    └── index.py        `python -m playlist_rag.cli.index`
```

## Stages

### Stage 1 — Normalize (`indexing/normalize.py`)

Pure functions, no I/O. Takes a parquet row (dict from
`pd.read_parquet().iterrows()`), produces a `NormalizedTrack` (Pydantic).

- `clean_lyrics(raw)` — NFKC normalize, strip Genius `[Verse]` annotations
  via regex, collapse 3+ newlines, rstrip each line. Empty result → `None`,
  not `""`.
- `_coerce_int / _coerce_float / _coerce_str / _coerce_list` — pandas
  returns numpy ints/floats and `NaN`; these helpers normalize to
  `int | float | str | list[str] | None`.
- `normalize_row(dict) → NormalizedTrack` — calls helpers, maps parquet
  columns to Pydantic fields.

### Stage 2 — Describe (`indexing/describe.py`)

LLM + `PydanticOutputParser` produces a `TrackSemantics` object — not free
text. Two prompt variants depending on whether the track has lyrics.

`TrackSemantics` schema (`schemas.py`):

```python
Mood = Literal["joyful", "melancholic", "angry", "calm", "energetic",
               "romantic", "introspective", "nostalgic", "rebellious", "uplifting"]
EnergyQualitative = Literal["very_low", "low", "medium", "high", "very_high"]

class TrackSemantics(BaseModel):
    description: str            # 30-600 chars, English prose
    themes: list[str]           # 1-5 short lowercase noun phrases
    mood: Mood                  # closed set
    inferred_subgenre: str      # free text, ≤50 chars
    energy_qualitative: EnergyQualitative  # closed set
```

Closed sets (`Mood`, `EnergyQualitative`) enforce vocabulary consistency for
filterable retrieval. Free text (`description`, `inferred_subgenre`)
preserves richness. `themes` is one-to-many for normalized filtering.

**Two prompt variants share one system prompt.** System prompt establishes
the role (music analyst), demands specificity, and includes audio-feature
interpretation guidance:

- high energy + high valence → upbeat, party, celebration
- high energy + low valence → angry, intense, aggressive
- low energy + high valence → calm, content, peaceful
- low energy + low valence → sad, melancholic, introspective
- high acousticness → organic, intimate, stripped-down
- high instrumentalness → ambient, focus, no vocals to follow
- high speechiness → spoken-word leaning, rap, talky
- high danceability + high tempo → club, propulsive
- low danceability + low tempo → slow-burn, ballad, contemplative

The two **user** prompt variants:

- **Full** (`has_lyrics=True`) — includes lyrics excerpt (truncated to
  `lyrics_excerpt_chars=800`), language, genius_tag.
- **Degraded** (`has_lyrics=False`, ~70% of catalog) — omits lyrics, adds an
  explicit "no lyrics — infer themes conservatively" note. Same
  `TrackSemantics` output.

**Retry policy** (`@retry` from tenacity):

- `OutputParserException` / `ValidationError` (malformed JSON, invalid enum)
  → retry once with exponential backoff, then re-raise.
- Transient API errors (rate-limit, 5xx) handled by tenacity too. LangChain's
  built-in `ChatOpenAI` retries are disabled (`max_retries=0`) — tenacity
  owns retries for explicit control.

### Stage 3 — Embed (`indexing/embed.py`)

OpenAI `text-embedding-3-small` via `langchain_openai.OpenAIEmbeddings`,
configured to emit 1536-D vectors. Dimension is asserted after every call so
misconfiguration fails fast.

No lazy loading or threading lock — the OpenAI client is cheap to instantiate,
and the embedding call is a network round-trip. (Earlier draft used local
bge-m3; see [decisions.md](decisions.md#embeddings-bge-m3--openai-text-embedding-3-small).)

### Stage 4 — Persist (`indexing/persist.py`)

Three operations per call, all in one SQLAlchemy session:

1. **Upsert `tracks`** — `INSERT ... ON CONFLICT (spotify_track_id) DO UPDATE`,
   `RETURNING id`. Updates everything except `id` and `spotify_track_id`.
2. **Upsert `track_embeddings`** — `ON CONFLICT (track_id) DO UPDATE` for
   `embedding`, `model_name`, `embedded_at`. Separate table so re-embedding
   is a `TRUNCATE` + re-run of stage 3 — `tracks` rows survive.
3. **Replace `track_themes`** — `DELETE WHERE track_id = ?`, then `INSERT`
   each theme. Themes per track are small (≤5), so delete-then-insert is
   simpler than a per-theme upsert.

Status helpers:

- `get_status(session, spotify_track_id)` — fetch one row.
- `mark_status(session, spotify_track_id, status, error=None)` — upsert into
  `track_index_status`. `attempt_count` increments only on `status='failed'`.
- `get_completed_ids(session)` — selects all `spotify_track_id` where
  `status='complete'`. Called **once at pipeline startup** to build the
  in-memory skip set.

### Orchestrator (`indexing/pipeline.py`)

```python
def run_pipeline(parquet_path, limit=None, retry_failed=False) -> RunStats:
    # 1. Build skip set ONCE (avoid per-track SELECT)
    with get_session() as session:
        completed = get_completed_ids(session)

    # 2. Stream parquet rows
    for row in tqdm(_iter_parquet(parquet_path, limit=limit)):
        if row.track_id in completed and not retry_failed:
            stats.skipped_complete += 1
            continue

        norm = normalize_row(row)        # stage 1
        ok, _ = _process_one(norm)        # stages 2–4, error-isolated
        stats.succeeded += 1 if ok else 0
        stats.failed    += 0 if ok else 1
```

`_process_one`:

- Calls `describe → embed_text → upsert_track → mark_status('complete')`
  inside one `get_session()`.
- On any exception: log, then in a **separate new session** call
  `mark_status('failed', error=str(e)[:500])`. Separate session because the
  original transaction is already rolled back.

`RunStats` dataclass tracks `total / skipped_complete / succeeded / failed /
elapsed_seconds` and renders a one-line summary with throughput.

## CLI

```bash
python -m playlist_rag.cli.index \
    --parquet data/unified_tracks.parquet \
    [--limit N]            # process at most N tracks (smoke test)
    [--retry-failed]       # re-attempt rows marked failed
    [-v]                   # DEBUG-level logging
```

Exit codes: `0` = no failures, `1` = parquet not found, `2` = at least one
track failed.

## Database schema

Four tables, all defined in `playlist_rag/db.py` (SQLAlchemy ORM) and created
by Alembic migration `0001_initial`.

### `tracks` — one row per track

All queryable scalar fields. Stable columns from the unified parquet plus the
LLM outputs (`description`, `mood`, `inferred_subgenre`,
`energy_qualitative`), tracking columns (`indexed_at`, `indexer_version`).

Indexes on filterable fields: `energy`, `tempo`, `valence`, `mood`,
`language`, `popularity_tier`. GIN index on `playlist_genres` (text array)
for genre filters.

### `track_embeddings` — one row per track

```sql
track_id      BIGINT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE
embedding     vector(1536) NOT NULL
model_name    TEXT NOT NULL
embedded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
```

Separate from `tracks` so re-embedding is `TRUNCATE track_embeddings;` + rerun
stage 3 — `tracks` rows untouched.

HNSW index on `embedding` is **deferred** to migration `0002_hnsw_index`.
See "HNSW deferral" below.

### `track_themes` — one-to-many

```sql
track_id  BIGINT REFERENCES tracks(id) ON DELETE CASCADE
theme     TEXT
PRIMARY KEY (track_id, theme)
INDEX (theme)
```

Normalized so queries like `SELECT * FROM tracks JOIN track_themes ON …
WHERE theme = 'heartbreak'` use the B-tree index. JSON array on `tracks`
would force GIN containment ops everywhere.

### `track_index_status` — resumability

```sql
spotify_track_id   TEXT PRIMARY KEY
status             TEXT NOT NULL    -- 'complete' | 'failed'
last_error         TEXT
last_attempt_at    TIMESTAMPTZ
attempt_count      INT NOT NULL DEFAULT 0
INDEX (status)
```

State machine:

- No row → never processed.
- `status='complete'` → skip on re-run.
- `status='failed'` → skip on re-run unless `--retry-failed`.

Intermediate per-stage statuses (`described`, `embedded`) are intentionally
not modeled — each track is processed atomically within one session. If
multi-stage failure recovery becomes useful later, they can be added without
breaking the existing state machine.

## HNSW deferral

`CREATE INDEX … USING hnsw` on an empty table while inserts arrive is 5-10×
slower than building the index once on a populated table.

Two migrations:

- `0001_initial.py` — all tables, all scalar indexes, **no HNSW**.
- `0002_hnsw_index.py` — only the HNSW index.

Workflow:

```
alembic upgrade 0001_initial          # create tables
python -m playlist_rag.cli.index ...  # bulk load
alembic upgrade head                  # build HNSW
```

Documented in `0002_hnsw_index.py`'s docstring so the next person doesn't
treat them as a single `alembic upgrade head`.

## Configuration

All env vars flow through `playlist_rag.config.settings` (pydantic-settings).
Single source of truth — no `os.getenv()` elsewhere in the codebase. See
`.env.example` for the full list. Notable fields:

| Field | Default | Notes |
|---|---|---|
| `database_url` | `postgresql://playlist:password@localhost:5435/playlist_rag` | SQLAlchemy URL |
| `openai_api_key` | `""` | Required at LLM-call time |
| `llm_model` | `gpt-4o-mini` | Swappable via LangChain |
| `llm_temperature` | `0.3` | |
| `embedding_model` | `text-embedding-3-small` | |
| `embedding_dim` | `1536` | Asserted after every call |
| `indexer_version` | `0.1.0` | Stored on each indexed row |
| `lyrics_excerpt_chars` | `800` | Truncation in full prompt |

## Error handling policy

| Failure | Handling |
|---|---|
| LLM 429 / 5xx | Tenacity retry, exponential backoff. Re-raise after attempts. |
| LLM malformed JSON / invalid enum | Tenacity retry once. Re-raise. |
| Embedding dim mismatch | `ValueError` — config error, fail fast. |
| Postgres connection drop | SQLAlchemy reconnect via `pool_pre_ping=True`. Per-track sessions limit blast radius. |
| Ctrl+C mid-run | `KeyboardInterrupt` propagates; current session rolls back. Next run picks up. |
| Lyrics encoding garbage | `clean_lyrics()` NFKC-normalizes. Empty result → degraded prompt. |
| Parquet column missing | `KeyError` from `normalize_row` — config error, fail loudly. |
| Per-track exception | Caught, logged via `logger.exception`, marked failed in a separate session, run continues. |

**Per-track error isolation is non-negotiable.** A 4,500-track run that fails
on track 100 must produce a database with rows 1-99 committed and row 100
recorded as failed.
