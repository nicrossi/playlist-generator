import argparse
import logging
import sys
from pathlib import Path

from playlist_rag.indexing.pipeline import run_pipeline


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="playlist_rag.cli.index",
        description="Index unified tracks parquet into Postgres + pgvector.",
    )
    parser.add_argument(
        "--parquet", required=True, type=Path, help="Path to unified_tracks.parquet"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N tracks (smoke test)"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt tracks marked failed",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="DEBUG-level logging"
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger("playlist_rag.cli.index")

    if not args.parquet.exists():
        log.error("Parquet file not found: %s", args.parquet)
        return 1

    stats = run_pipeline(
        args.parquet, limit=args.limit, retry_failed=args.retry_failed
    )
    log.info("Done. %s", stats)
    print(stats)
    return 2 if stats.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
