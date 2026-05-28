from pydantic import BaseModel, Field

from playlist_rag.schemas import GenerationRun


class EvalCase(BaseModel):

    id: str
    query: str
    reference_statements: list[str] = Field(
        default_factory=list,
        description=(
            "Rubric requirements the retrieved context should support (RAGAS-style "
            "context recall). Each statement is checked against retrieved tracks."
        ),
    )
    notes: str | None = None


class QueryMetrics(BaseModel):
    case_id: str
    query: str
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    duration_adherence: float | None = None
    exclusion_adherence: float | None = None
    artist_diversity: float | None = None
    genre_diversity: float | None = None
    retrieved_count: int = 0
    playlist_count: int = 0
    retrieval_relaxed: bool = False
    judge_notes: dict[str, str] = Field(default_factory=dict)


class EvalSummary(BaseModel):
    cases: int = 0
    context_precision_mean: float | None = None
    context_recall_mean: float | None = None
    faithfulness_mean: float | None = None
    answer_relevance_mean: float | None = None
    duration_adherence_mean: float | None = None
    exclusion_adherence_mean: float | None = None
    artist_diversity_mean: float | None = None
    genre_diversity_mean: float | None = None


class EvalReport(BaseModel):
    summary: EvalSummary
    per_query: list[QueryMetrics]
    runs: list[GenerationRun] = Field(
        default_factory=list,
        description="Full runs when include_runs=True",
    )
