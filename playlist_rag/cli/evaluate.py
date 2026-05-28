import argparse
import json
import logging
import sys
from pathlib import Path

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_summary(report) -> None:
    s = report.summary
    print("\n=== Evaluation summary ===")
    print(f"  cases:              {s.cases}")
    print(f"  context_precision:  {_fmt(s.context_precision_mean)}")
    print(f"  context_recall:     {_fmt(s.context_recall_mean)}")
    print(f"  faithfulness:       {_fmt(s.faithfulness_mean)}")
    print(f"  answer_relevance:   {_fmt(s.answer_relevance_mean)}")
    print(f"  duration_adherence: {_fmt(s.duration_adherence_mean)}")
    print(f"  exclusion_adherence:{_fmt(s.exclusion_adherence_mean)}")
    print(f"  artist_diversity:   {_fmt(s.artist_diversity_mean)}")
    print(f"  genre_diversity:    {_fmt(s.genre_diversity_mean)}")
    print("\nPer query:")
    for q in report.per_query:
        print(
            f"  [{q.case_id}] prec={_fmt(q.context_precision)} "
            f"rec={_fmt(q.context_recall)} "
            f"faith={_fmt(q.faithfulness)} "
            f"ans={_fmt(q.answer_relevance)} "
            f"dur={_fmt(q.duration_adherence)} "
            f"excl={_fmt(q.exclusion_adherence)} "
            f"art-div={_fmt(q.artist_diversity)} "
            f"gen-div={_fmt(q.genre_diversity)} "
            f"(retrieved={q.retrieved_count}, playlist={q.playlist_count})"
        )
    print()


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="playlist_rag.cli.evaluate",
        description=(
            "Evaluate playlist generation with RAG metrics: "
            "context precision/recall, faithfulness, answer relevance."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/queries.jsonl"),
        help="JSON or JSONL file of EvalCase objects",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write full report JSON")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N cases only")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval pool size")
    parser.add_argument(
        "--no-llm-explain",
        action="store_true",
        help="Template explanation during generation (cheaper)",
    )
    parser.add_argument(
        "--skip-retrieval-judge",
        action="store_true",
        help="Skip retrieval LLM judges (context precision + context recall)",
    )
    parser.add_argument(
        "--skip-generation-judge",
        action="store_true",
        help="Skip faithfulness and answer relevance judges",
    )
    parser.add_argument(
        "--include-runs",
        action="store_true",
        help="Embed full GenerationRun objects in output JSON",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    if not args.dataset.exists():
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    try:
        from playlist_rag.eval.dataset import load_eval_cases

        cases = load_eval_cases(args.dataset)
        if args.limit is not None:
            cases = cases[: args.limit]

        from playlist_rag.eval.runner import run_evaluation

        report = run_evaluation(
            cases,
            top_k=args.top_k,
            use_llm_explain=not args.no_llm_explain,
            skip_retrieval_judge=args.skip_retrieval_judge,
            skip_generation_judge=args.skip_generation_judge,
            include_runs=args.include_runs,
        )
    except Exception as e:
        logging.getLogger("playlist_rag.cli.evaluate").exception("Evaluation failed")
        print(f"error: {e}", file=sys.stderr)
        return 1

    _print_summary(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = report.model_dump(mode="json")
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
