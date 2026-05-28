# Playlist Generator

A retrieval-augmented playlist generator over a unified catalog of
Spotify + Genius tracks. The system takes a natural-language query
("melancholic late-night drives", "upbeat songs for cleaning the apartment")
and returns a sequenced playlist with per-track explanations.

The repo covers **data ingestion, indexing, and playlist generation** from
natural-language queries.

## Components

| Component | What it does | Deep dive |
|---|---|---|
| `download_top50.py` | Pulls Kaggle Top-50 dataset, resolves each row via YouTube + `yt-dlp`, saves WAVs. | [`docs/download_top50.md`](docs/download_top50.md) |
| `extract_features.py` | Spotify-compatible audio features from local WAV (Essentia + MusiCNN). | [`docs/extract_features.md`](docs/extract_features.md) |
| `unify.py` | Fuzzy-joins two Spotify CSVs with the Genius lyrics CSV into one parquet catalog. | [`docs/dataset_unification.md`](docs/dataset_unification.md) |
| `playlist_rag/` | Indexes the unified parquet → LLM description + embedding → Postgres + pgvector. | [`docs/indexing_pipeline.md`](docs/indexing_pipeline.md) |
| `playlist_rag.cli.generate` | NL query → hybrid retrieval → ranked, sequenced playlist + explanation. | [`docs/generation_pipeline.md`](docs/generation_pipeline.md) |
| `playlist_rag.cli.evaluate` | RAG metrics: context precision/recall, faithfulness, answer relevance. | [`docs/evaluation.md`](docs/evaluation.md) |
| Design log | Why each component is shaped the way it is. | [`docs/decisions.md`](docs/decisions.md) |

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` covers all components (audio extraction + indexing). The
audio path has heavy ML dependencies (`essentia-tensorflow`); skip them if
you only need the indexing pipeline.

## End-to-end quickstart

### 1. Get the data

```bash
# Kaggle datasets (popular + obscure Spotify + Genius lyrics) → data/
```

### 2. Unify Spotify + Genius

```bash
python unify.py \
    --spotify-popular data/high_popularity_spotify_data.csv \
    --spotify-obscure data/low_popularity_spotify_data.csv \
    --genius data/genius_song_lyrics.csv \
    --out data/unified_tracks.parquet \
    --report-dir reports/
```

First run ~10 min (Genius stream), cached after.

### 3. (Optional) Backfill missing lyrics

For tracks without a Genius match (~70% of the catalog), fetch lyrics from
[lyrics.ovh](https://github.com/NTag/lyrics.ovh) without re-running `unify.py`:

```bash
python -m playlist_rag.cli.backfill_lyrics --limit 20 --backup
python -m playlist_rag.cli.backfill_lyrics --backup --clear-index-status
python -m playlist_rag.cli.index --parquet data/unified_tracks.parquet
```

See [`docs/lyrics_backfill.md`](docs/lyrics_backfill.md).

### 4. Index into Postgres

```bash
docker compose up -d                    # Postgres 16 + pgvector
cp .env.example .env                    # set OPENAI_API_KEY
alembic upgrade 0001_initial            # create tables (no HNSW yet)
python -m playlist_rag.cli.index \
    --parquet data/unified_tracks.parquet \
    --limit 20                          # smoke test on 20 tracks
python -m playlist_rag.cli.index \
    --parquet data/unified_tracks.parquet
alembic upgrade head                    # build HNSW index after bulk load
```

See [`docs/indexing_pipeline.md`](docs/indexing_pipeline.md) for the full
indexer reference (stages, schema, resumability, error handling).

### 5. Extract features for tracks outside the dataset

```bash
python extract_features.py tracks/top50/*.wav --csv --output top50_features.csv
```

See [`docs/extract_features.md`](docs/extract_features.md).

### 6. Generate a playlist

```bash
python -m playlist_rag.cli.generate "música tranquila para estudiar 2 horas sin reggaetón"
python -m playlist_rag.cli.generate "upbeat rock for a workout" --json
```

Requires indexed data, HNSW migration (`alembic upgrade head`), and `OPENAI_API_KEY` in `.env`.

See [`docs/generation_pipeline.md`](docs/generation_pipeline.md) for stages,
schemas, filters, sequencing, and CLI reference.

### 7. Evaluate (optional)

```bash
python -m playlist_rag.cli.evaluate --dataset eval/queries.jsonl --limit 1
```

See [`docs/evaluation.md`](docs/evaluation.md) for metrics, dataset format, and cost notes.

### 8. Launch the web UI

```bash
streamlit run playlist_rag/ui/app.py
```

A Streamlit front end over the same pipeline: enter a query, get the
explanation, the sequenced playlist (with Spotify links), a panel showing what
the model parsed, and the retrieval pool. Same prerequisites as the CLI
(indexed data, `alembic upgrade head`, `OPENAI_API_KEY` in `.env`).

## Repository layout

```
playlist-generator/
├── README.md                  # this file
├── docs/                      # per-component deep dives + decisions log
├── requirements.txt           # all deps (audio + indexing)
├── docker-compose.yml         # Postgres 16 + pgvector
├── alembic.ini, alembic/      # database migrations
├── .env.example               # config template
│
├── download_top50.py          # WAV downloader
├── extract_features.py        # Essentia/MusiCNN audio feature extractor
├── unify.py                   # Spotify + Genius unification
├── playlist_rag/              # indexing pipeline Python package
│   ├── config.py, schemas.py, db.py
│   ├── indexing/              # 4 stages: normalize, describe, embed, persist
│   ├── query/                 # NL → QueryIntent
│   ├── retrieval/             # hybrid vector + SQL search
│   ├── playlist/              # rank, sequence, explain
│   ├── eval/                  # RAG evaluation metrics
│   ├── cli/
│   │   ├── index.py
│   │   ├── generate.py
│   │   └── evaluate.py
│   └── ui/                    # Streamlit web app (app.py)
│
├── data/                      # input CSVs + unified parquet (gitignored)
├── tracks/                    # downloaded WAVs (gitignored)
├── models/                    # cached Essentia/MusiCNN TF models
└── reports/                   # unification match reports
```

## Status

- ✅ Data ingestion (Spotify CSVs, Genius lyrics, Kaggle Top-50 WAVs)
- ✅ Local audio feature extraction (Essentia adapter)
- ✅ Dataset unification (fuzzy Spotify ↔ Genius join)
- ✅ Indexing pipeline (LLM description + embedding → Postgres + pgvector)
- ✅ Retrieval engine (hybrid SQL + vector search)
- ✅ Query rewriter / intent parser
- ✅ Playlist sequencing + per-track explanation
- ✅ End-user CLI (`python -m playlist_rag.cli.generate`)
- ✅ RAG evaluation (`python -m playlist_rag.cli.evaluate`)
- ✅ Web UI (`streamlit run playlist_rag/ui/app.py`)
