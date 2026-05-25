# Design Decisions

A running log of the design calls made along the way, with the rationale
behind each. 

---

## Feature extraction

### Essentia over Web API

**Chose:** Extract audio features locally via Essentia + MusiCNN.

**Why:** Spotify deprecated the `audio_features` Web API endpoint in
November 2024. Anything outside the pre-collected Kaggle dataset has no
upstream source for the per-track attributes the recommender uses (energy,
danceability, valence, tempo, key, etc.).

**Rejected:** Building the recommender purely on the Kaggle dataset. That
freezes the catalog and prevents adding new tracks. The Essentia adapter
lets the rest of the pipeline treat Kaggle-sourced and locally-extracted
tracks uniformly through the same Spotify schema.

### Spotify-compatible output schema

**Chose:** `extract_features.py` emits exactly Spotify's `audio_features`
column set (plus a `file_path` leading column in CSV mode for validation).

**Why:** Downstream code (unifier, indexer, retrieval) doesn't need to
branch on data source. One schema, one normalization path.

**Caveat:** Two fields don't come from audio: `liveness` defaults to `0.0`
(studio-recorded majority) and `time_signature` defaults to `4` (4/4
majority). Query rewriters should treat filters on those fields as no-ops
for Essentia-path tracks.

### Heuristic `energy` blend

**Chose:** `energy = 0.5 * loudness_norm + 0.3 * rms_norm + 0.2 * flux_norm`.

**Why:** Spotify's `energy` is not a single algorithm — it's a blend that
correlates with loudness, perceived intensity, and spectral richness. We
approximate with three measurable proxies: loudness (volume), RMS (signal
power), spectral flux (timbral change rate). Weights are tunable against
the Spotify reference column via the `--validate` mode.

### Speechiness as `voice_prob * (1 - key_confidence)`

**Chose:** Multiply the voice/instrumental classifier's voice probability by
`1 - key_confidence`.

**Why:** Spoken word usually means high vocal presence with weak tonal
structure (no clear key). A song that's just vocal-forward but sung in a
key (rap with a beat, ballad) shouldn't read as high-speechiness — the
`(1 - key_confidence)` term suppresses that. Heuristic, gets evaluated by
the validation harness.

---

## Dataset unification

### Fuzzy match on canonical keys

**Chose:** Canonicalize `(artist, title)` (NFKD, lowercase, strip suffixes),
then exact match → fuzzy title within artist → fuzzy both.
`rapidfuzz.token_sort_ratio` thresholds: artist ≥ 90, title ≥ 88.

**Why:** No shared primary key between Spotify and Genius. Direct equality
misses ~half the catalog because of suffix variations (`- Remastered 2011`,
`(feat. ...)`, `- Live`). Fuzzy with bounded thresholds recovers them
without over-matching.

**Tuning loop:** `reports/{summary,fuzzy_sample,unmatched_sample}.csv` are
the feedback signal. Manually inspect → adjust suffix patterns or
thresholds → re-run.

### Stream the 8.6 GB Genius CSV

**Chose:** Never load Genius as a DataFrame. Stream once, build a compact
in-memory probe index, pickle it.

**Why:** Pandas would balloon to ~25 GB RAM on the full Genius CSV.
Streaming keeps peak memory under 2 GB. The pickle cache means subsequent
runs skip the streaming step and finish in under a minute.

### Left join (keep unmatched Spotify tracks)

**Chose:** Spotify tracks without a Genius hit are kept with null lyrics.

**Why:** ~70% of tracks have no Genius match — overwhelmingly because the
title doesn't canonicalize to anything in Genius (non-English scripts,
release-year differences). Throwing them out would gut the catalog. The
indexing pipeline's degraded prompt handles them downstream.

---

## Indexing pipeline (`playlist_rag`)


### Universal indexing with degraded prompt

**Chose:** Generate descriptions for **all** tracks. Two prompt variants —
full (lyrics + features + metadata) and degraded (features + metadata only,
explicit "no lyrics" note). Both produce the same `TrackSemantics` output.

**Why:** ~30% of tracks have lyrics. Restricting to those would gut the
catalog. The degraded variant gives the LLM enough signal (artist, genre
tags, audio features) to produce something useful — and the prompt warns it
to "infer themes conservatively" so the model doesn't hallucinate themes
without lyrics to ground them.

**Rejected:** Skipping no-lyrics tracks, or generating only features-based
descriptions for them. Both reduce catalog coverage; neither materially
improves quality of the lyric'd subset.

### Cloud LLM (GPT-4o-mini) over local

**Chose:** OpenAI `gpt-4o-mini` via LangChain's `ChatOpenAI`.

**Why:** ~4,500 tracks × ~600 tokens per call ≈ $1.50 total. Local Llama
adds 8+ GB model download, GPU dependency, slower iteration. The cost
difference is rounding error at this scale.

**Provider-swappable:** LangChain abstraction means changing to Claude or
local Llama is a config swap, not a rewrite. But default stays cloud unless
we have a reason.

### Structured LLM output via Pydantic

**Chose:** Output is a `TrackSemantics` Pydantic model — not free text.
LangChain's `PydanticOutputParser` enforces the schema.

**Why:** Free text would force downstream code to parse / regex the
description for filterable fields (mood, energy level). Pydantic validation
at the LLM boundary catches malformed outputs as exceptions, which the
retry policy handles cleanly.

**Closed sets** (`Mood`, `EnergyQualitative` as `Literal`) enforce
vocabulary consistency so filters like `WHERE mood = 'melancholic'` work
without normalization mapping. Free text (`description`, `inferred_subgenre`)
preserves richness for the fields that don't benefit from a closed set.

### Embeddings: bge-m3 → OpenAI text-embedding-3-small

**Chose:** OpenAI `text-embedding-3-small` (1536-D).

**Originally chose** bge-m3 (1024-D, local via HuggingFace). Switched after
re-examining the tradeoffs at our scale.

**Why the switch:**

- **Cost is rounding error.** ~4,500 tracks × ~100 tokens per description ≈
  450K tokens. At $0.02/1M, total embedding cost is < $0.01. Indexing
  goes from "$1.50" to "$1.51".
- **Stack simplification.** Dropping `langchain-huggingface` and
  `sentence-transformers` removes ~3 GB of install size (torch) and a 2 GB
  model download. `embed.py` shrinks from ~50 lines (lazy singleton +
  threading lock) to ~30 (network call).
- **Same provider as the LLM.** Already have OpenAI configured. One vendor
  story for indexing.

**Tradeoffs:**

- Network dependency for indexing. But the LLM step also needs network, so
  the failure profile doesn't change.
- Provider lock-in. If OpenAI deprecates or hikes pricing, we re-embed.
  At our scale, that's a 1-minute operation.
- Slight loss of "fully open-source RAG" story for portfolio purposes.
  Acceptable.

**bge-m3's actual advantages** (multilingual edge, no provider dependency)
don't apply here: descriptions are English-only, and the catalog is too
small for re-embedding cost to matter.

### Per-track sessions, not one transaction

**Chose:** One `get_session()` per track. Commits after each track.

**Why:** A 4,500-track run takes ~1-2 hours. Wrapping it in one transaction
means a failure on track 100 rolls back tracks 1-99 — losing hours of work
and dollars of LLM calls. Per-track sessions commit incrementally; failure
isolates to one row.

**Cost:** More round-trips to Postgres. At ~1-2 seconds per track (LLM is
the bottleneck), the round-trip overhead is invisible. If batching ever
matters, `batch_commit_size` is reserved in `settings`.

### Idempotency via `ON CONFLICT DO UPDATE`

**Chose:** PostgreSQL upsert for `tracks` and `track_embeddings`.
Delete-then-insert for `track_themes`.

**Why:** Re-running indexing produces the same database state — no
duplicates, latest values win. Crucial for the resumability story: a
mid-run kill, restart, and "did we process this track?" check has to be
trivial.

**Themes use delete-then-insert** because the set may shrink (e.g., LLM
produced 4 themes the first time, 2 themes after a prompt tweak).
On-conflict won't remove orphaned rows; explicit delete-then-insert does.

### Resumability via `track_index_status` + in-memory skip set

**Chose:** A separate `track_index_status` table tracks per-track state
(`complete` / `failed`). At pipeline startup, **one query** populates a
`set[str]` of completed IDs; the hot loop skips via in-memory membership.

**Why:** Per-track SELECT-before-process would double the DB round-trips.
The single startup query is O(n) but happens once. The hot loop's
"is this track done?" check is now O(1) in memory.

**State machine kept simple:** no `described`, `embedded` intermediate
states. If a stage fails mid-track, the whole track is `failed`; the next
run retries the whole track. Multi-stage recovery is added complexity we
don't need yet.

### Hierarchical Navigable Small World (HNSW) index built after bulk load (separate migration)

**Chose:** Two Alembic migrations. `0001_initial` creates all tables and
scalar indexes **but not** the HNSW index. `0002_hnsw_index` adds it
afterward.

**Why:** Building HNSW incrementally during inserts is 5-10× slower than
building it once on a populated table. The workflow:

```
alembic upgrade 0001_initial          # create tables
python -m playlist_rag.cli.index ...  # bulk load
alembic upgrade head                  # build HNSW
```

### pgvector over a dedicated vector DB

**Chose:** Postgres + pgvector. Single database for structured filtering
and vector search.

**Why:** Joins between vector results and scalar filters work natively
(`WHERE mood = 'sad' ORDER BY embedding <=> query_vec`). 

**Rejected:** Pinecone, and others. All would force a dual-database
architecture.

### Three Postgres tables, not one

**Chose:** `tracks` (scalar fields), `track_embeddings` (vectors),
`track_themes` (one-to-many), `track_index_status` (state).

**Why:**

- **Separating embeddings** means re-embedding is `TRUNCATE
  track_embeddings;` + rerun stage 3. The `tracks` rows survive — including
  the (expensive) LLM-generated descriptions.
- **Normalizing themes** lets filters use a B-tree index on `theme` instead
  of GIN containment ops on a JSON column.
- **State as a separate table** means clearing the indexing progress
  (`TRUNCATE track_index_status;`) doesn't touch the catalog data.

### LangChain only at the LLM and embedding boundaries

**Chose:** Use LangChain for two things: `ChatOpenAI + ChatPromptTemplate +
PydanticOutputParser` in `describe.py`, and `OpenAIEmbeddings` in
`embed.py`. Everything else is plain Python + SQLAlchemy + pandas.

**Why:** LangChain is a useful vendor-agnostic wrapper for those specific
calls. It is a bad abstraction for the orchestration loop, retrieval
engine, or sequencing — those need explicit control we'd fight LangChain
to maintain.

**No agents, memory, chains beyond `prompt | llm | parser`.** The pipeline
is a deterministic function loop. Agent abstractions add debugging surface
area without benefit when the flow is known.

### Settings singleton via pydantic-settings

**Chose:** A single `settings = Settings()` in `playlist_rag/config.py`.
**No `os.getenv()` elsewhere.**

**Why:** Type safety, defaults, `.env` loading, and validation at import
time — all for one dependency. The "single source of truth" rule means
configuration drift is impossible: there's no second place to check.

### Tenacity for retries, not LangChain's built-in

**Chose:** `ChatOpenAI(max_retries=0)` + `@retry` wrapping `describe()`.

**Why:** Explicit control over which exceptions retry, how many times, and
with what backoff. LangChain's retry behavior is opaque from the call
site. Tenacity's `retry_if_exception_type` makes the policy readable and
testable.

### Pydantic everywhere

**Chose:** Pydantic at all data-shape boundaries — LLM output
(`TrackSemantics`), inter-stage payload (`NormalizedTrack`), and even the
LLM parser. SQLAlchemy ORM handles DB I/O (no separate Pydantic model
needed for that).

**Why:** Three boundaries, three validation points. Typos and shape drift
become exceptions at construction time, not silent data corruption miles
downstream.
