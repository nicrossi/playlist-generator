import logging
from pathlib import Path

from tqdm import tqdm

from playlist_rag.eval.dataset import load_eval_cases
from playlist_rag.eval.metrics import compute_query_metrics
from playlist_rag.eval.schemas import EvalCase, EvalReport, EvalSummary, QueryMetrics
from playlist_rag.playlist.pipeline import generate_playlist_with_trace
from playlist_rag.schemas import GenerationRun

logger = logging.getLogger(__name__)


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _summarize(per_query: list[QueryMetrics]) -> EvalSummary:
    return EvalSummary(
        cases=len(per_query),
        context_precision_mean=_mean([q.context_precision for q in per_query]),
        context_recall_mean=_mean([q.context_recall for q in per_query]),
        faithfulness_mean=_mean([q.faithfulness for q in per_query]),
        answer_relevance_mean=_mean([q.answer_relevance for q in per_query]),
        duration_adherence_mean=_mean([q.duration_adherence for q in per_query]),
        exclusion_adherence_mean=_mean([q.exclusion_adherence for q in per_query]),
        artist_diversity_mean=_mean([q.artist_diversity for q in per_query]),
        genre_diversity_mean=_mean([q.genre_diversity for q in per_query]),
    )


def run_evaluation(
    cases: list[EvalCase],
    *,
    top_k: int | None = None,
    use_llm_explain: bool = True,
    skip_retrieval_judge: bool = False,
    skip_generation_judge: bool = False,
    include_runs: bool = False,
) -> EvalReport:
    per_query: list[QueryMetrics] = []
    runs: list[GenerationRun] = []

    for case in tqdm(cases, desc="Eval", unit="query"):
        logger.info("Generating playlist for case %s", case.id)
        run = generate_playlist_with_trace(
            case.query,
            use_llm_explain=use_llm_explain,
            top_k=top_k,
        )
        if include_runs:
            runs.append(run)

        metrics = compute_query_metrics(
            case,
            run,
            skip_retrieval_judge=skip_retrieval_judge,
            skip_generation_judge=skip_generation_judge,
        )
        per_query.append(metrics)

    return EvalReport(
        summary=_summarize(per_query),
        per_query=per_query,
        runs=runs,
    )


def run_evaluation_from_file(
    dataset_path: Path,
    **kwargs,
) -> EvalReport:
    cases = load_eval_cases(dataset_path)
    if not cases:
        raise ValueError(f"no eval cases in {dataset_path}")
    return run_evaluation(cases, **kwargs)
