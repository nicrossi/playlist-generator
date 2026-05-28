import logging

from playlist_rag.config import settings
from playlist_rag.eval.context import format_playlist_answer
from playlist_rag.eval.judge import (
    judge_answer_relevance,
    judge_faithfulness,
    judge_statement_coverage_batch,
    judge_track_relevance_batch,
)
from playlist_rag.eval.objective import (
    artist_diversity,
    duration_adherence,
    exclusion_adherence,
    genre_diversity,
)
from playlist_rag.eval.schemas import EvalCase, QueryMetrics
from playlist_rag.schemas import GenerationRun, RetrievedTrack

logger = logging.getLogger(__name__)


def _chunk_tracks(
    tracks: list[RetrievedTrack], batch_size: int
) -> list[list[RetrievedTrack]]:
    return [tracks[i : i + batch_size] for i in range(0, len(tracks), batch_size)]


def _judge_relevance_map(
    query: str, tracks: list[RetrievedTrack]
) -> dict[str, bool]:
   
    relevance: dict[str, bool] = {}
    batch_size = settings.eval_judge_batch_size
    for batch in _chunk_tracks(tracks, batch_size):
        result = judge_track_relevance_batch(query, batch)
        for j in result.judgments:
            relevance[j.spotify_track_id] = j.relevant
        for t in batch:
            if t.spotify_track_id not in relevance:
                relevance[t.spotify_track_id] = False
    return relevance


def _ragas_context_precision_at_k(relevance_by_rank: list[bool]) -> float:
    
    if not relevance_by_rank:
        return 0.0
    true_positives_at_k = 0
    weighted_sum = 0.0
    for k, is_relevant in enumerate(relevance_by_rank, start=1):
        if is_relevant:
            true_positives_at_k += 1
            precision_at_k = true_positives_at_k / k
            weighted_sum += precision_at_k
    total_relevant = true_positives_at_k
    if total_relevant == 0:
        return 0.0
    return weighted_sum / total_relevant


def context_precision(
    query: str, retrieved: list[RetrievedTrack]
) -> tuple[float | None, str]:
    
    if not retrieved:
        return None, "no retrieved tracks"
    to_judge = retrieved[: settings.eval_max_retrieved_to_judge]
    rel_map = _judge_relevance_map(query, to_judge)
    relevance_by_rank = [
        rel_map.get(t.spotify_track_id, False) for t in to_judge
    ]
    k = len(relevance_by_rank)
    precision = _ragas_context_precision_at_k(relevance_by_rank)
    relevant_count = sum(relevance_by_rank)
    note = (
        f"RAGAS Context Precision@{k}: {relevant_count}/{k} relevant "
        f"(judged {k}/{len(retrieved)} retrieved)"
    )
    return precision, note


def context_recall(
    query: str,
    retrieved: list[RetrievedTrack],
    reference_statements: list[str],
) -> tuple[float | None, str]:
    
    statements = [s.strip() for s in reference_statements if s.strip()]
    if not statements:
        return None, "no reference_statements"
    if not retrieved:
        return 0.0, "no retrieved tracks"

    contexts = retrieved[: settings.eval_max_retrieved_to_judge]
    result = judge_statement_coverage_batch(query, statements, contexts)
    covered_by_index: dict[int, bool] = {}
    for j in result.judgments:
        if 0 <= j.statement_index < len(statements):
            covered_by_index[j.statement_index] = j.covered

    covered_count = sum(
        1 for i in range(len(statements)) if covered_by_index.get(i, False)
    )
    recall = covered_count / len(statements)
    note = (
        f"covered {covered_count}/{len(statements)} statements "
        f"over {len(contexts)}/{len(retrieved)} retrieved tracks"
    )
    return recall, note


def compute_query_metrics(
    case: EvalCase,
    run: GenerationRun,
    *,
    skip_retrieval_judge: bool = False,
    skip_generation_judge: bool = False,
) -> QueryMetrics:
    result = run.result
    retrieved = run.trace.retrieved_candidates
    metrics = QueryMetrics(
        case_id=case.id,
        query=case.query,
        retrieved_count=len(retrieved),
        playlist_count=len(result.tracks),
        retrieval_relaxed=run.trace.retrieval_relaxed,
    )

    # Deterministic constraint + diversity metrics: no LLM, always computed.
    try:
        metrics.duration_adherence, dur_note = duration_adherence(
            result.intent, result.total_duration_minutes
        )
        metrics.judge_notes["duration_adherence"] = dur_note
        metrics.exclusion_adherence, excl_note = exclusion_adherence(
            result.intent, result.tracks
        )
        metrics.judge_notes["exclusion_adherence"] = excl_note
        metrics.artist_diversity, art_note = artist_diversity(result.tracks)
        metrics.judge_notes["artist_diversity"] = art_note
        metrics.genre_diversity, gen_note = genre_diversity(result.tracks)
        metrics.judge_notes["genre_diversity"] = gen_note
    except Exception as e:
        logger.exception("objective metrics failed for %s", case.id)
        metrics.judge_notes["objective_metrics_error"] = str(e)

    if not skip_retrieval_judge and retrieved:
        prec, note = context_precision(case.query, retrieved)
        metrics.context_precision = prec
        metrics.judge_notes["context_precision"] = note

    if not skip_retrieval_judge and case.reference_statements:
        try:
            rec, note = context_recall(
                case.query, retrieved, case.reference_statements
            )
            metrics.context_recall = rec
            metrics.judge_notes["context_recall"] = note
        except Exception as e:
            logger.exception("context recall judge failed for %s", case.id)
            metrics.judge_notes["context_recall_error"] = str(e)

    if skip_generation_judge:
        return metrics

    playlist_context = result.tracks if result.tracks else retrieved
    if result.explanation and playlist_context:
        try:
            faith = judge_faithfulness(
                case.query, playlist_context, result.explanation
            )
            metrics.faithfulness = faith.score
            metrics.judge_notes["faithfulness"] = (
                f"RAGAS: {faith.supported_claims}/{faith.total_claims} claims "
                f"attributable. {faith.reasoning}"
            )
            if faith.claims:
                metrics.judge_notes["faithfulness_claims"] = " | ".join(
                    faith.claims
                )
        except Exception as e:
            logger.exception("faithfulness judge failed for %s", case.id)
            metrics.judge_notes["faithfulness_error"] = str(e)

    if result.tracks or result.explanation:
        try:
            answer_block = format_playlist_answer(
                case.query,
                result.tracks,
                result.explanation,
                result.total_duration_minutes,
            )
            rel = judge_answer_relevance(case.query, answer_block)
            metrics.answer_relevance = rel.score
            metrics.judge_notes["answer_relevance"] = rel.reasoning
            if rel.generated_questions:
                metrics.judge_notes["answer_relevance_questions"] = " | ".join(
                    rel.generated_questions
                )
        except Exception as e:
            logger.exception("answer_relevance judge failed for %s", case.id)
            metrics.judge_notes["answer_relevance_error"] = str(e)

    return metrics
