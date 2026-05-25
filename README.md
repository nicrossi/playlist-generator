# Playlist Generator

A retrieval-augmented playlist generator over a unified catalog of
Spotify + Genius tracks. The system takes a natural-language query
("melancholic late-night drives", "upbeat songs for cleaning the apartment")
and returns a sequenced playlist with per-track explanations.

This repo currently covers the **data ingestion + indexing** half of the
system. Retrieval, query rewriting, and sequencing are the next stages.

## Components

| Component | What it does | Deep dive |
|---|---|---|
| `download_top50.py` | Pulls Kaggle Top-50 dataset, resolves each row via YouTube + `yt-dlp`, saves WAVs. | [`docs/download_top50.md`](docs/download_top50.md) |
| `extract_features.py` | Spotify-compatible audio features from local WAV (Essentia + MusiCNN). | [`docs/extract_features.md`](docs/extract_features.md) |
| `unify.py` | Fuzzy-joins two Spotify CSVs with the Genius lyrics CSV into one parquet catalog. | [`docs/dataset_unification.md`](docs/dataset_unification.md) |
| `playlist_rag/` | Indexes the unified parquet → LLM description + embedding → Postgres + pgvector. | [`docs/indexing_pipeline.md`](docs/indexing_pipeline.md) |
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

### 3. Index into Postgres

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

### 4. Extract features for tracks outside the dataset

```bash
python extract_features.py tracks/top50/*.wav --csv --output top50_features.csv
```

See [`docs/extract_features.md`](docs/extract_features.md).

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
│   └── cli/index.py
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
- ⬜ Retrieval engine (hybrid SQL + vector search)
- ⬜ Query rewriter / intent parser
- ⬜ Playlist sequencing + per-track explanation
- ⬜ End-user CLI
