"""
Unify Spotify (popular + obscure) with Genius lyrics into one DataFrame.

Pipeline:
  1. Load both Spotify CSVs, add popularity_tier, concat
  2. Dedupe by track_id, aggregate playlist info into arrays
  3. Build canonical match keys (artist, title)
  4. Stream Genius CSV once, build in-memory probe index
  5. For each Spotify track, probe Genius (exact then fuzzy)
  6. Write unified parquet + match report

Usage:
    python unify.py \\
        --spotify-popular data/high_popularity_spotify_data.csv \\
        --spotify-obscure data/low_popularity_spotify_data.csv \\
        --genius data/genius_song_lyrics.csv \\
        --out unified_tracks.parquet \\
        --report-dir reports/

Dependencies:
    pip install pandas pyarrow rapidfuzz
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process


# Columns taken first-value when deduping Spotify by track_id (stable per-track).
SPOTIFY_STABLE_COLS = [
    "energy", "tempo", "danceability", "loudness", "liveness", "valence",
    "track_artist", "time_signature", "speechiness", "track_popularity",
    "track_name", "track_album_name", "track_album_release_date",
    "instrumentalness", "track_album_id", "mode", "key", "duration_ms",
    "acousticness",
]

# Columns aggregated into arrays (one row per playlist appearance).
SPOTIFY_PLAYLIST_COLS = ["playlist_id", "playlist_name", "playlist_genre", "playlist_subgenre"]

# Genius columns kept in the probe index; rest discarded to save RAM.
GENIUS_KEEP_COLS = ["lyrics", "tag", "language", "year", "id"]

# Title suffixes that differ between Spotify and Genius for the same recording.
TITLE_SUFFIX_PATTERNS = [
    r"\s*-\s*Remaster(ed)?.*$",
    r"\s*-\s*\d{4}\s+Remaster.*$",
    r"\s*-\s*Live.*$",
    r"\s*-\s*Acoustic.*$",
    r"\s*-\s*Radio Edit.*$",
    r"\s*-\s*From\s+.*$",
    r"\s*-\s*Single Version.*$",
    r"\s*-\s*Album Version.*$",
    r"\s*\(Remaster(ed)?.*\)$",
    r"\s*\(Live.*\)$",
    r"\s*\(Acoustic.*\)$",
    r"\s*\(feat\.?\s+.*\)$",
    r"\s*\(with\s+.*\)$",
    r"\s*\(From\s+.*\)$",
]
_TITLE_SUFFIX_RE = re.compile("|".join(TITLE_SUFFIX_PATTERNS), re.IGNORECASE)

# rapidfuzz token_sort_ratio thresholds (0-100).
ARTIST_FUZZY_THRESHOLD = 90
TITLE_FUZZY_THRESHOLD = 88


def canonicalize(text: str) -> str:
    """Lowercase, strip accents, drop punctuation, collapse whitespace."""
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[.\-_'\"\!\?\,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_title_suffixes(title: str) -> str:
    """Strip '- Remastered 2011', '(Live)', etc."""
    if not isinstance(title, str):
        return ""
    return _TITLE_SUFFIX_RE.sub("", title).strip()


def extract_primary_artist(artists_field: str) -> str:
    """'Lady Gaga, Bruno Mars' -> 'Lady Gaga'."""
    if not isinstance(artists_field, str) or not artists_field:
        return ""
    return artists_field.split(",")[0].strip()


def load_and_concat_spotify(popular_csv: Path, obscure_csv: Path) -> pd.DataFrame:
    """Load both Spotify CSVs, tag with popularity_tier, concat."""
    print(f"Loading Spotify popular: {popular_csv}")
    df_pop = pd.read_csv(popular_csv)
    df_pop["popularity_tier"] = "popular"

    print(f"Loading Spotify obscure: {obscure_csv}")
    df_obs = pd.read_csv(obscure_csv)
    df_obs["popularity_tier"] = "obscure"

    df = pd.concat([df_pop, df_obs], ignore_index=True)
    print(f"  Combined rows: {len(df):,} ({len(df_pop):,} popular + {len(df_obs):,} obscure)")
    return df


def dedupe_spotify(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe by track_id; stable cols take first value, playlist cols aggregate to sorted unique arrays."""
    agg_dict: dict[str, Any] = {col: (col, "first") for col in SPOTIFY_STABLE_COLS}
    agg_dict["popularity_tier"] = ("popularity_tier", "first")
    for col in SPOTIFY_PLAYLIST_COLS:
        agg_dict[f"{col}s"] = (col, lambda x: sorted(set(x.dropna().astype(str))))

    deduped = df.groupby("track_id", as_index=False).agg(**agg_dict)

    print(f"  Deduped to {len(deduped):,} unique tracks (was {len(df):,})")
    print(f"  Tier breakdown: {dict(deduped['popularity_tier'].value_counts())}")
    return deduped


def add_match_keys_spotify(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical artist/title columns for joining against Genius."""
    return df.assign(
        _artist_primary=df["track_artist"].apply(extract_primary_artist),
        _title_clean=df["track_name"].apply(strip_title_suffixes),
    ).assign(
        _artist_key=lambda d: d["_artist_primary"].apply(canonicalize),
        _title_key=lambda d: d["_title_clean"].apply(canonicalize),
    )


def build_genius_index(genius_csv: Path) -> tuple[dict, dict]:
    """
    Stream Genius CSV once, build two indexes:
      exact_index:  (artist_key, title_key) -> slim row dict
      by_artist:    artist_key -> [(title_key, slim_row), ...]

    For ~5M rows expect ~5-10 minutes and ~2-3 GB RAM.
    """
    exact_index: dict[tuple[str, str], dict] = {}
    by_artist: dict[str, list] = defaultdict(list)

    start = perf_counter()
    with genius_csv.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 250_000 == 0 and i > 0:
                elapsed = perf_counter() - start
                print(f"  indexed {i:,} Genius rows ({elapsed:.0f}s elapsed, "
                      f"{i / elapsed:,.0f} rows/sec)")

            artist_raw = row.get("artist", "")
            title_raw = row.get("title", "")
            if not artist_raw or not title_raw:
                continue

            artist_key = canonicalize(artist_raw)
            title_key = canonicalize(strip_title_suffixes(title_raw))
            if not artist_key or not title_key:
                continue

            slim = {k: row.get(k, "") for k in GENIUS_KEEP_COLS}
            exact_index[(artist_key, title_key)] = slim
            by_artist[artist_key].append((title_key, slim))

    elapsed = perf_counter() - start
    print(f"Indexed {len(exact_index):,} (artist, title) keys across "
          f"{len(by_artist):,} artists in {elapsed:.0f}s")
    return exact_index, by_artist


def load_or_build_genius_index(genius_csv: Path, cache_path: Path) -> tuple[dict, dict]:
    """Cache the Genius index to avoid re-streaming 8.4 GB every run."""
    if cache_path.exists():
        print(f"Loading cached Genius index: {cache_path}")
        start = perf_counter()
        with cache_path.open("rb") as f:
            exact_index, by_artist = pickle.load(f)
        print(f"  Loaded in {perf_counter() - start:.1f}s "
              f"({len(exact_index):,} keys, {len(by_artist):,} artists)")
        return exact_index, by_artist

    exact_index, by_artist = build_genius_index(genius_csv)
    print(f"Caching index to: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump((exact_index, by_artist), f, protocol=pickle.HIGHEST_PROTOCOL)
    return exact_index, by_artist


def probe_genius(
    artist_key: str,
    title_key: str,
    exact_index: dict,
    by_artist: dict,
) -> tuple[dict | None, str]:
    """
    Three-tier probe:
      1. Exact (artist, title)
      2. Exact artist + fuzzy title within that artist
      3. Fuzzy artist + fuzzy title

    Returns (matched_row_or_None, match_type).
    """
    if (artist_key, title_key) in exact_index:
        return exact_index[(artist_key, title_key)], "exact"

    if artist_key in by_artist:
        titles = [t for t, _ in by_artist[artist_key]]
        match = process.extractOne(
            title_key, titles,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=TITLE_FUZZY_THRESHOLD,
        )
        if match:
            _, _, idx = match
            return by_artist[artist_key][idx][1], "fuzzy_title"

    artist_match = process.extractOne(
        artist_key, list(by_artist.keys()),
        scorer=fuzz.token_sort_ratio,
        score_cutoff=ARTIST_FUZZY_THRESHOLD,
    )
    if artist_match:
        matched_artist, _, _ = artist_match
        titles = [t for t, _ in by_artist[matched_artist]]
        title_match = process.extractOne(
            title_key, titles,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=TITLE_FUZZY_THRESHOLD,
        )
        if title_match:
            _, _, idx = title_match
            return by_artist[matched_artist][idx][1], "fuzzy_both"

    return None, "unmatched"


def join_spotify_to_genius(
    spotify_df: pd.DataFrame, exact_index: dict, by_artist: dict
) -> pd.DataFrame:
    """Left join: every Spotify row preserved, Genius fields nullable."""
    print(f"Joining {len(spotify_df):,} Spotify tracks against Genius...")
    results = []
    match_types: Counter[str] = Counter()
    start = perf_counter()

    for i, (_, sp_row) in enumerate(spotify_df.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  joined {i:,}/{len(spotify_df):,}")

        genius_row, match_type = probe_genius(
            sp_row["_artist_key"], sp_row["_title_key"], exact_index, by_artist
        )
        match_types[match_type] += 1

        out = sp_row.to_dict()
        out["_match_type"] = match_type
        if genius_row:
            out["lyrics"] = genius_row["lyrics"]
            out["genius_tag"] = genius_row["tag"]
            out["language"] = genius_row["language"]
            out["genius_year"] = genius_row["year"]
            out["genius_id"] = genius_row["id"]
        else:
            out["lyrics"] = None
            out["genius_tag"] = None
            out["language"] = None
            out["genius_year"] = None
            out["genius_id"] = None
        results.append(out)

    print(f"  Joined in {perf_counter() - start:.1f}s")
    print(f"  Match types: {dict(match_types)}")
    return pd.DataFrame(results)


def write_match_report(joined: pd.DataFrame, report_dir: Path) -> None:
    """Write artifacts for manual review of join quality."""
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = joined["_match_type"].value_counts().to_dict()
    total = len(joined)
    matched = total - counts.get("unmatched", 0)

    summary = {
        "total_tracks": total,
        "exact_matches": counts.get("exact", 0),
        "fuzzy_title_matches": counts.get("fuzzy_title", 0),
        "fuzzy_both_matches": counts.get("fuzzy_both", 0),
        "unmatched": counts.get("unmatched", 0),
        "match_rate_pct": round(100 * matched / total, 1) if total else 0,
    }
    pd.Series(summary).to_frame("value").to_csv(report_dir / "summary.csv")

    fuzzy = joined[joined["_match_type"].isin(["fuzzy_title", "fuzzy_both"])]
    if not fuzzy.empty:
        fuzzy[["track_artist", "track_name", "_match_type", "lyrics"]].head(100).to_csv(
            report_dir / "fuzzy_sample.csv", index=False
        )

    unmatched = joined[joined["_match_type"] == "unmatched"]
    if not unmatched.empty:
        unmatched[["track_artist", "track_name", "popularity_tier", "track_popularity"]].head(100).to_csv(
            report_dir / "unmatched_sample.csv", index=False
        )

    print("\n=== Match Report ===")
    print(f"  Total tracks:     {summary['total_tracks']:,}")
    print(f"  Exact matches:    {summary['exact_matches']:,}")
    print(f"  Fuzzy (title):    {summary['fuzzy_title_matches']:,}")
    print(f"  Fuzzy (both):     {summary['fuzzy_both_matches']:,}")
    print(f"  Unmatched:        {summary['unmatched']:,}")
    print(f"  Match rate:       {summary['match_rate_pct']}%")
    print(f"  Artifacts:        {report_dir}/")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--spotify-popular", type=Path, required=True)
    p.add_argument("--spotify-obscure", type=Path, required=True)
    p.add_argument("--genius", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output parquet path")
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument(
        "--genius-cache", type=Path, default=Path(".cache/genius_index.pkl"),
        help="Pickle cache for the Genius index (~3 GB; speeds up reruns)",
    )
    p.add_argument(
        "--rebuild-cache", action="store_true",
        help="Force rebuild of the Genius index cache",
    )
    args = p.parse_args()

    for path, name in [
        (args.spotify_popular, "spotify-popular"),
        (args.spotify_obscure, "spotify-obscure"),
        (args.genius, "genius"),
    ]:
        if not path.exists():
            print(f"error: --{name} file not found: {path}", file=sys.stderr)
            return 1

    if args.rebuild_cache and args.genius_cache.exists():
        args.genius_cache.unlink()

    print("=== Step 1: Spotify ===")
    spotify = load_and_concat_spotify(args.spotify_popular, args.spotify_obscure)
    spotify = dedupe_spotify(spotify)
    spotify = add_match_keys_spotify(spotify)

    print("\n=== Step 2: Genius Index ===")
    exact_index, by_artist = load_or_build_genius_index(args.genius, args.genius_cache)

    print("\n=== Step 3: Join ===")
    joined = join_spotify_to_genius(spotify, exact_index, by_artist)

    joined = joined.drop(
        columns=[c for c in joined.columns if c.startswith("_") and c != "_match_type"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.out, index=False)
    print(f"\nWrote unified parquet: {args.out} ({len(joined):,} rows)")

    write_match_report(joined, args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
