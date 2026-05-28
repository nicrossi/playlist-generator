# Lyrics backfill (`lyrics.ovh`)

Optional second source for tracks that have no Genius match in `unify.py`
(~70% of the catalog). Uses the public API documented in
[lyrics.ovh](https://github.com/NTag/lyrics.ovh):

```http
GET https://api.lyrics.ovh/v1/{artist}/{title}
```

This does **not** re-run `unify.py` or the full indexing job. Only rows with
empty `lyrics` are fetched. Tracks that already have lyrics are untouched.

## Quickstart

```bash
# Smoke test (no API, no write)
python -m playlist_rag.cli.backfill_lyrics --dry-run --limit 5

# Try 20 tracks
python -m playlist_rag.cli.backfill_lyrics --limit 20 --backup

# Full backfill (~3135 tracks, ~40–90 min with default delay)
python -m playlist_rag.cli.backfill_lyrics --backup --clear-index-status

# Selective re-index (skips tracks still marked complete)
python -m playlist_rag.cli.index --parquet data/unified_tracks.parquet
```

## What gets written

| Output | Purpose |
|--------|---------|
| `data/unified_tracks.parquet` | `lyrics` + `lyrics_source='lyrics.ovh'` on success |
| `data/lyrics_cache/*.json` | Resume cache (gitignored under `data/`) |
| `reports/reindex_track_ids.txt` | Spotify IDs that gained lyrics |
| `reports/lyrics_backfill_summary.json` | Counts (found / not found / errors) |

## Selective re-index

Indexing skips rows in `track_index_status` with `status='complete'`.
Use `--clear-index-status` during backfill to delete status **only** for
tracks that received new lyrics, then run `cli.index` again.

Cost is proportional to **newly backfilled count** (LLM describe + embed each),
not the full ~4500-track run.

## Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--parquet` | `data/unified_tracks.parquet` | Input catalog |
| `--out` | overwrite input | Output path |
| `--backup` | off | Copy parquet to `.bak` before overwrite |
| `--cache-dir` | `data/lyrics_cache` | Per-track JSON cache |
| `--delay` | `0.75` | Seconds between API calls |
| `--limit N` | all missing | Cap attempts |
| `--dry-run` | off | Count only |
| `--no-cache` | off | Force fresh API calls |
| `--clear-index-status` | off | Queue IDs for re-index |
| `--reindex-ids-out` | `reports/reindex_track_ids.txt` | ID manifest |

## Title normalization

The client tries stripped titles (no `- Remaster`, `(feat. …)`, etc.) before
the raw Spotify title, using the same helpers as `unify.py`.

## Limitations

- API availability and rate limits are not guaranteed; use `--delay`.
- Many obscure tracks will still return 404.
- Re-index only needed for tracks that gained lyrics if you want richer
  `description` / embeddings (full LLM prompt with lyrics excerpt).
