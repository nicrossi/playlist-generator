import argparse
import logging
import shutil
import sys
from pathlib import Path

from playlist_rag.ingest.backfill import (
    backfill_parquet_lyrics,
    clear_index_status_for_reindex,
    write_reindex_manifest,
)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="playlist_rag.cli.backfill_lyrics",
        description=(
            "Backfill missing lyrics in unified_tracks.parquet using "
            "https://api.lyrics.ovh (no full re-unify or full re-index)."
        ),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/unified_tracks.parquet"),
        help="Input unified parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet (default: overwrite --parquet in place)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy input parquet to <parquet>.bak before overwriting",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/lyrics_cache"),
        help="Per-track JSON cache for resume",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help="Seconds to wait after each API request",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N tracks missing lyrics",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count candidates only; no API calls or parquet write",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cache files",
    )
    parser.add_argument(
        "--clear-index-status",
        action="store_true",
        help="Delete track_index_status for backfilled IDs (enables selective re-index)",
    )
    parser.add_argument(
        "--reindex-ids-out",
        type=Path,
        default=Path("reports/reindex_track_ids.txt"),
        help="Write spotify track_ids that gained lyrics",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("reports/lyrics_backfill_summary.json"),
        help="Write JSON summary of the backfill run",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger("playlist_rag.cli.backfill_lyrics")

    if not args.parquet.exists():
        log.error("Parquet not found: %s", args.parquet)
        return 1

    out_path = args.out
    if out_path is None and not args.dry_run:
        if args.backup:
            backup = Path(f"{args.parquet}.bak")
            shutil.copy2(args.parquet, backup)
            log.info("Backup written to %s", backup)
        out_path = args.parquet

    try:
        stats = backfill_parquet_lyrics(
            args.parquet,
            out_path=out_path,
            cache_dir=args.cache_dir,
            delay_seconds=args.delay,
            limit=args.limit,
            dry_run=args.dry_run,
            use_cache=not args.no_cache,
        )
    except Exception as e:
        log.exception("Backfill failed")
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not args.dry_run:
        if args.summary_out:
            write_reindex_manifest(
                stats.backfilled_ids,
                args.reindex_ids_out,
                args.summary_out,
                stats,
            )
        if stats.backfilled_ids:
            log.info(
                "Re-index manifest: %s (%d ids)",
                args.reindex_ids_out,
                len(stats.backfilled_ids),
            )

        if args.clear_index_status and stats.backfilled_ids:
            removed = clear_index_status_for_reindex(stats.backfilled_ids)
            log.info(
                "Cleared track_index_status for %d rows (re-run cli.index next)",
                removed,
            )

    print("\n=== Lyrics backfill summary ===")
    print(f"  candidates (no lyrics):  {stats.candidates}")
    print(f"  already had lyrics:     {stats.skipped_has_lyrics}")
    print(f"  api attempts:           {stats.fetched}")
    print(f"  cache hits:             {stats.cache_hits}")
    print(f"  found:                  {stats.found}")
    print(f"  not found:              {stats.not_found}")
    print(f"  errors:                 {stats.errors}")
    if stats.backfilled_ids:
        print(f"  reindex ids file:       {args.reindex_ids_out}")
    print()
    if stats.found and not args.dry_run:
        print(
            "Next step (selective re-index, skips other complete tracks):\n"
            f"  python -m playlist_rag.cli.index --parquet {out_path or args.parquet}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
