# `unify.py` — Dataset Unification

Joins two Spotify Kaggle CSVs (popular + obscure) with the Genius lyrics CSV
into a single parquet catalog. Spotify provides audio features + metadata;
Genius provides lyrics + language + tag. There is no shared primary key, so
the join is fuzzy on `(artist, track_title)` after canonicalization.

Output: `unified_tracks.parquet` — the input the indexing pipeline reads from.

## Why fuzzy

Spotify and Genius name the same recording differently:

- `"Bohemian Rhapsody - Remastered 2011"` (Spotify) vs `"Bohemian Rhapsody"` (Genius)
- `"Locked Out of Heaven (feat. ...)"` vs `"Locked Out of Heaven"`
- `"Niño con Acentos"` vs `"Nino con Acentos"` (accent stripping)
- Trailing tags like `- Live`, `- Acoustic`, `- Radio Edit`, `(From "Frozen")`

A direct equality join misses all of these. Canonicalization + bounded fuzzy
matching recovers them without over-matching unrelated tracks.

## Pipeline

1. **Load Spotify CSVs.** Two files (`high_popularity_spotify_data.csv`,
   `low_popularity_spotify_data.csv`). Add `popularity_tier` (`popular` /
   `obscure`), concatenate.
2. **Dedupe by `track_id`.** Many rows per track because tracks appear in
   multiple playlists. Per-track-stable columns (audio features, name,
   artist) take the first value. Playlist columns
   (`playlist_id/name/genre/subgenre`) aggregate into arrays.
3. **Build canonical match keys.** Lowercase, NFKD-strip accents, drop
   parenthetical featured-artist tags, strip suffix patterns (Remastered,
   Live, Radio Edit, From "...", etc.). Result: `(artist_canon, title_canon)`
   per row.
4. **Stream Genius CSV.** It is 8.6 GB — never loaded as a DataFrame. Stream
   it once, dropping columns we don't need (`GENIUS_KEEP_COLS = lyrics, tag,
   language, year, id`). Build a compact in-memory probe index keyed by
   `(artist_canon, title_canon)`, plus a per-artist title list for the fuzzy
   fallback.
5. **Probe per Spotify track.** Try exact-key hit → fuzzy title within
   exact-artist bucket → fuzzy both. `rapidfuzz.fuzz.token_sort_ratio`
   thresholds: artist ≥ 90, title ≥ 88. First match wins. Tracks without a
   Genius hit are kept with null lyrics (left join).
6. **Write parquet + match report.** Output parquet to `--out`. Report CSVs
   to `--report-dir`: `summary.csv` (per-match-type counts), `fuzzy_sample.csv`
   (random sample of fuzzy hits for spot-checking), `unmatched_sample.csv`
   (random sample of misses for tuning suffixes / thresholds).

## Canonicalization rules

`canonicalize()` applies, in order:

1. NFKD normalize, drop combining marks (strip accents).
2. Lowercase.
3. Strip suffix patterns from `TITLE_SUFFIX_PATTERNS` (regex alternation).
4. Collapse whitespace.

Suffix patterns to be aware of when adding new sources:

```
- Remaster[ed] …
- <year> Remaster …
- Live … / (Live …)
- Acoustic … / (Acoustic …)
- Radio Edit …
- From "…"
- Single Version …
- Album Version …
(feat. …)
(with …)
```

If the unmatched-sample report shows a recurring suffix not in this list, add
it and re-run with `--rebuild-cache`.

## Cache

The Genius probe index is pickled to `.cache/genius_index.pkl` after the
first build. Subsequent runs hit the pickle and skip the streaming step.

- First run: ~10 minutes (Genius stream + index build).
- Subsequent runs: < 1 minute.
- `--rebuild-cache` forces a re-stream — required after editing
  canonicalization rules or `GENIUS_KEEP_COLS`.

## Usage

```bash
python unify.py \
    --spotify-popular data/high_popularity_spotify_data.csv \
    --spotify-obscure data/low_popularity_spotify_data.csv \
    --genius data/genius_song_lyrics.csv \
    --out data/unified_tracks.parquet \
    --report-dir reports/
```

## Tuning the join

The match report at `reports/summary.csv` is the feedback signal. Per-tier
hit rates and per-match-type counts (`exact`, `fuzzy_title`, `fuzzy_both`,
`unmatched`) reveal where canonicalization is leaking.

- If `fuzzy_sample.csv` shows false positives (different songs matched), the
  thresholds (`ARTIST_FUZZY_THRESHOLD=90`, `TITLE_FUZZY_THRESHOLD=88`) are
  too loose — raise them.
- If `unmatched_sample.csv` shows recoverable misses with a common suffix
  pattern, add the pattern to `TITLE_SUFFIX_PATTERNS`.
- If `unmatched_sample.csv` shows misses in a non-English script (Japanese,
  Korean, Cyrillic), the canonicalization isn't going to help — those rows
  are kept with null lyrics, which is fine for the indexing pipeline's
  degraded-prompt path.

## Output schema

The parquet has all per-track-stable Spotify columns + aggregated playlist
arrays + Genius columns + `popularity_tier` + `_match_type` (`exact` /
`fuzzy_title` / `fuzzy_both` / `unmatched`).

Roughly 4,500 rows, ~30% with non-null lyrics. See
[indexing_pipeline.md](indexing_pipeline.md) for how downstream consumes it.
