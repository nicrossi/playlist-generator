#!/usr/bin/env python3
"""Glue the fresh-track path into one command: download -> extract -> unify -> index.

Each step is an existing CLI; this just chains them with a shared slug/track_id
contract and stops on the first failure. Steps are independently re-runnable, and
indexing is idempotent (already-indexed track_ids are skipped), so re-running only
processes genuinely new tracks.

Default mode is fresh-only incremental: unify writes a small parquet of just the
new Essentia tracks (no base-catalog concat, no Genius re-stream), and the indexer
embeds only track_ids not already complete. Safe to run nightly. Lyrics for fresh
tracks come from the separate lyrics.ovh backfill, not Genius.

Usage (nightly incremental):
    python ingest_fresh.py --out data/fresh_unified.parquet --limit 50

Usage (full catalog rebuild):
    python ingest_fresh.py --full-rebuild \\
        --spotify-popular data/high_popularity_spotify_data.csv \\
        --spotify-obscure data/low_popularity_spotify_data.csv \\
        --genius data/genius_song_lyrics.csv \\
        --out data/unified_tracks.parquet \\
        --limit 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(step: str, cmd: list[str]) -> None:
    print(f"\n=== {step} ===\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run([sys.executable, *cmd])
    if result.returncode != 0:
        sys.exit(f"error: step '{step}' failed (exit {result.returncode})")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default=None, help="Kaggle dataset slug for download_top50")
    p.add_argument("--output-dir", type=Path, default=Path("tracks/top50"))
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--cookies-from-browser", default=None)
    p.add_argument("--no-spotify", action="store_true")
    p.add_argument("--features-out", type=Path, default=Path("data/fresh_features.csv"))
    p.add_argument("--spotify-popular", type=Path)
    p.add_argument("--spotify-obscure", type=Path)
    p.add_argument("--genius", type=Path)
    p.add_argument("--out", type=Path, required=True, help="Unified parquet path")
    p.add_argument(
        "--full-rebuild", action="store_true",
        help="Rebuild the whole catalog parquet (concat base CSVs + Genius join). "
             "Default is fresh-only incremental: index just the new tracks.",
    )
    args = p.parse_args()

    if args.full_rebuild:
        missing = [n for n, v in (("--spotify-popular", args.spotify_popular),
                                  ("--spotify-obscure", args.spotify_obscure),
                                  ("--genius", args.genius)) if v is None]
        if missing:
            sys.exit(f"error: --full-rebuild requires {', '.join(missing)}")

    manifest = args.output_dir / "manifest.csv"
    args.features_out.parent.mkdir(parents=True, exist_ok=True)

    download = ["download_top50.py", "--output-dir", str(args.output_dir),
                "--limit", str(args.limit), "--manifest", str(manifest)]
    if args.dataset:
        download += ["--dataset", args.dataset]
    if args.cookies_from_browser:
        download += ["--cookies-from-browser", args.cookies_from_browser]
    if args.no_spotify:
        download.append("--no-spotify")
    run("download", download)

    wavs = sorted(str(p) for p in args.output_dir.glob("*.wav"))
    if not wavs:
        sys.exit(f"error: no WAVs in {args.output_dir}")
    run("extract", ["extract_features.py", *wavs, "--manifest", str(manifest),
                    "--csv", "--output", str(args.features_out)])

    if args.full_rebuild:
        run("unify", ["unify.py", "--spotify-popular", str(args.spotify_popular),
                      "--spotify-obscure", str(args.spotify_obscure),
                      "--genius", str(args.genius), "--spotify-fresh", str(args.features_out),
                      "--out", str(args.out)])
    else:
        run("unify", ["unify.py", "--fresh-only",
                      "--spotify-fresh", str(args.features_out), "--out", str(args.out)])

    run("index", ["-m", "playlist_rag.cli.index", "--parquet", str(args.out)])

    print("\nfresh ingestion complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
