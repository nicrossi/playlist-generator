# `download_top50.py`

Downloads the Spotify Top-50 Kaggle dataset's tracks as WAV files by searching YouTube via `yt-dlp`.

## What it does

1. Pulls the Kaggle dataset (default `anxods/spotify-top-50-playlist-songs-anxods`) into a temp dir and locates the first CSV.
2. Reads each row, picking the track / artist / duration columns by fuzzy header match (`track_name|song|name|title`, `artists|artist|track_artist`, `duration_ms|duration|track_duration_ms`).
3. For each `(artist, title)` pair, downloads the best YouTube match as a WAV into `tracks/top50/<slug>.wav`.
4. Prints a per-track status line and a final `downloaded / skipped / failed` summary.

## How a track is picked

Each track goes through three phases:

- **Phase A — search.** `ytsearch5:<artist> <title> official audio` pulls metadata for 5 candidates (no download).
- **Phase B — score.** Each candidate is scored by `|candidate_duration_ms − expected_ms|`; the closest wins. Candidates with unknown duration score `inf`. If the dataset has no duration column, all scores tie at `0` → first hit wins.
- **Phase C — download.** Winner's `webpage_url` is fetched and passed through `FFmpegExtractAudio` (codec `wav`).

If the target WAV already exists the track is skipped (idempotent re-runs).

## Auth

Kaggle credentials are read from env vars:

```
KAGGLE_USERNAME=<your-username>
KAGGLE_API_TOKEN=<your-api-token>
```

The script copies `KAGGLE_API_TOKEN` into `KAGGLE_KEY` for the Kaggle SDK.

## CLI

```
python download_top50.py [--dataset SLUG] [--output-dir PATH] [--limit N]
                         [--dry-run] [--cookies-from-browser BROWSER]
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--dataset` | `anxods/spotify-top-50-playlist-songs-anxods` | Kaggle dataset slug |
| `--output-dir` | `tracks/top50` | Where WAVs are written |
| `--limit` | `50` | Cap on tracks read from the CSV |
| `--dry-run` | off | Print queries + target slugs, no download |
| `--cookies-from-browser` | none | Source cookies from a browser (`chrome`, `firefox`, `safari`, …) to bypass YouTube's bot-check |

## Output naming

Slug is `slugify(f"{artist}_{title}")` — lowercased, non-alphanumerics collapsed to `_`. Final path: `<output-dir>/<slug>.wav`.

## Exit code

Returns `1` if any track failed, `0` otherwise.
