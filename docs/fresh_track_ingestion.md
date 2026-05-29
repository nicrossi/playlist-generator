# Fresh-track ingestion

Adds tracks that are **not** in the Kaggle Spotify catalog: download audio,
extract Spotify-compatible features locally (Essentia), give them a real Spotify
`track_id`, and fold them into the Postgres + pgvector index alongside everything
else.

Two modes:

- **Fresh-only (default)** — writes a small parquet of *just* the new Essentia
  tracks and indexes those. No base-catalog concat, no 8.4 GB Genius re-stream.
  Fast and idempotent → safe to run **nightly**. Lyrics for fresh tracks come
  from the separate lyrics.ovh backfill, not Genius.
- **Full rebuild (`--full-rebuild`)** — regenerates the whole
  `unified_tracks.parquet` from the base CSVs + Genius join (the original
  behaviour). Use when the base catalog or Genius data changes.

Both paths converge on the same index, which keys on `track_id` and skips
already-complete tracks, so neither duplicates existing data.

Before this path existed, `download_top50.py` and `extract_features.py` produced
schema-compatible data but shared no key — the extracted CSV had an empty
`track_id` and `track_artist`, and the indexer keys on `track_id`
(`playlist_rag/indexing/pipeline.py`). Fresh tracks would collide on `""`. The
glue below propagates identity end to end via the WAV **slug**.

## One command

Nightly incremental (default):

```bash
python ingest_fresh.py --out data/fresh_unified.parquet --limit 50
```

Full catalog rebuild:

```bash
python ingest_fresh.py --full-rebuild \
    --spotify-popular data/high_popularity_spotify_data.csv \
    --spotify-obscure data/low_popularity_spotify_data.csv \
    --genius data/genius_song_lyrics.csv \
    --out data/unified_tracks.parquet \
    --limit 50
```

`--spotify-popular`, `--spotify-obscure`, and `--genius` are only used (and
required) with `--full-rebuild`; the default fresh-only path ignores them.

Runs four steps in order, stopping on the first failure:

```
download_top50.py   -> tracks/top50/<slug>.wav  +  tracks/top50/manifest.csv
extract_features.py -> data/fresh_features.csv   (identity stamped from manifest)
unify.py            -> parquet   (fresh-only: new tracks, no Genius join;
                                  full-rebuild: whole catalog + Genius lyrics)
playlist_rag.cli.index -> Postgres  (idempotent: only new track_ids indexed)
```

## The slug contract

The join key across every stage is `slug = slugify(f"{artist}_{title}")` — the
WAV filename. `download_top50.py` writes it into `manifest.csv`;
`extract_features.py` recovers it from the WAV stem (`Path(source_path).stem`)
and copies the manifest's identity columns (`track_id`, `track_artist`,
`track_name`, album fields, popularity) onto each feature row.

`manifest.csv` columns: `slug, track_id, track_artist, track_name,
track_album_name, track_album_id, track_album_release_date, track_popularity,
duration_ms, id_source, status`.

## Track IDs

`download_top50.py` resolves the **real** Spotify `track_id` via the Spotify Web
API search endpoint (client-credentials flow — no user login). Set credentials
in `.env`:

```
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
```

(Spotify deprecated the audio-features endpoint in Nov 2024, which is why we
still extract features locally; search remains available for identity.)

Matching uses a rapidfuzz `token_sort_ratio` guard (threshold 80) on
artist+title to reject bad hits. On a miss — or with `--no-spotify`, or no
credentials — the track gets a deterministic synthetic id `fresh:<slug>` so the
pipeline always completes. `id_source` records which.

## Dedupe conflict policy

**Full rebuild:** `unify.py` concatenates fresh rows **last** (popular → obscure →
fresh), so `dedupe_spotify`'s `"first"` aggregation keeps the canonical Kaggle row
whenever a fresh `track_id` already exists. Fresh rows therefore only *add*
genuinely new tracks; they never overwrite existing Kaggle audio features with
Essentia ones. (Overriding is intentionally out of scope.) Fresh rows flow through
the normal Genius fuzzy join, picking up lyrics whenever artist+title match.

**Fresh-only:** no base catalog in the parquet, so dedupe runs over fresh rows
alone. The "don't overwrite existing tracks" guarantee instead comes from the
indexer: `get_completed_ids` skips any `track_id` already complete
(`playlist_rag/indexing/pipeline.py`), and `upsert_track` is an idempotent
`ON CONFLICT DO UPDATE` (`playlist_rag/indexing/persist.py`). No Genius join runs;
rows are tagged `_match_type="fresh_no_genius"` with null lyrics, to be filled by
the lyrics.ovh backfill.

### Lyrics for fresh tracks

Genius is skipped on the fresh-only path (new/obscure songs are rarely in the
Genius dump, and re-streaming it nightly is wasteful). After ingestion, run the
lyrics.ovh backfill to populate lyrics for the new `track_id`s, then re-index
those ids.

## Running steps individually

The orchestrator is a convenience; each step is a standalone CLI and
independently re-runnable.

```bash
python download_top50.py --limit 50 --output-dir tracks/top50          # + manifest.csv
python extract_features.py tracks/top50/*.wav \
    --manifest tracks/top50/manifest.csv --csv --output data/fresh_features.csv
# fresh-only (default path):
python unify.py --fresh-only \
    --spotify-fresh data/fresh_features.csv --out data/fresh_unified.parquet
python -m playlist_rag.cli.index --parquet data/fresh_unified.parquet

# full rebuild instead:
python unify.py --spotify-popular ... --spotify-obscure ... --genius ... \
    --spotify-fresh data/fresh_features.csv --out data/unified_tracks.parquet
python -m playlist_rag.cli.index --parquet data/unified_tracks.parquet
```

`extract_features.py` without `--manifest` is unchanged (empty identity columns),
so the tool still works standalone for ad-hoc feature extraction.
