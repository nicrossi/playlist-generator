import argparse
import json
import logging
import sys

from playlist_rag.playlist.pipeline import generate_playlist


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_playlist(result) -> None:
    print(f"\n{result.explanation}\n")
    print(
        f"Playlist ({len(result.tracks)} tracks, "
        f"~{result.total_duration_minutes:.0f} min)\n"
    )
    for t in result.tracks:
        dur_min = (t.duration_ms or 0) / 60_000
        print(f"  {t.position:2d}. {t.track_name} — {t.track_artist} ({dur_min:.1f}m)")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="playlist_rag.cli.generate",
        description="Generate a playlist from a natural-language query.",
    )
    parser.add_argument("query", help="Natural-language playlist request")
    parser.add_argument(
        "--json", action="store_true", help="Emit full PlaylistResult as JSON"
    )
    parser.add_argument(
        "--no-llm-explain",
        action="store_true",
        help="Use template explanation instead of LLM",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval pool size (default from settings)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger("playlist_rag.cli.generate")

    try:
        result = generate_playlist(
            args.query,
            use_llm_explain=not args.no_llm_explain,
            top_k=args.top_k,
        )
    except Exception as e:
        log.exception("Playlist generation failed")
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_playlist(result)

    return 0 if result.tracks else 2


if __name__ == "__main__":
    sys.exit(main())
