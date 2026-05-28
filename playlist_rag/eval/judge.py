import numpy as np
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from playlist_rag.config import settings
from playlist_rag.eval.context import format_context_block, format_track_context
from playlist_rag.indexing.embed import embed_batch, embed_text
from playlist_rag.schemas import PlaylistTrack, RetrievedTrack

_llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=settings.eval_judge_temperature,
    timeout=settings.llm_timeout_seconds,
    max_retries=2,
    api_key=settings.openai_api_key or None,
)


class TrackRelevanceJudgment(BaseModel):
    spotify_track_id: str
    relevant: bool


class TrackRelevanceBatch(BaseModel):
    judgments: list[TrackRelevanceJudgment]


class StatementCoverageJudgment(BaseModel):
    statement_index: int = Field(ge=0)
    covered: bool


class StatementCoverageBatch(BaseModel):
    judgments: list[StatementCoverageJudgment]


class ExtractedClaims(BaseModel):
    claims: list[str] = Field(
        default_factory=list,
        description="Atomic factual claims from the generated explanation.",
    )


class ClaimAttributionJudgment(BaseModel):
    claim_index: int = Field(ge=0)
    attributable: bool = Field(
        description="True if the claim can be inferred from the track contexts alone."
    )


class ClaimAttributionBatch(BaseModel):
    judgments: list[ClaimAttributionJudgment]


class FaithfulnessResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    total_claims: int = Field(ge=0)
    supported_claims: int = Field(ge=0)
    claims: list[str] = Field(default_factory=list)
    reasoning: str = ""


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        description=(
            "Paraphrased playlist requests the answer satisfies "
            "(same intent as a user query, not meta-questions about the response)."
        ),
    )


class AnswerRelevanceResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    num_questions: int = Field(ge=0)
    generated_questions: list[str] = Field(default_factory=list)
    cosine_similarities: list[float] = Field(default_factory=list)
    reasoning: str = ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


_RELEVANCE_SYSTEM = """You judge whether retrieved music tracks are relevant to a user's playlist request.
A track is relevant if its mood, genre, energy, and description reasonably match what the user asked for.
Be strict on explicit exclusions (e.g. "no reggaeton" → reggaeton tracks are NOT relevant).
{format_instructions}"""

_RELEVANCE_USER = """User request:
{query}

Tracks to judge (mark each spotify_track_id as relevant true/false):
{tracks_block}
"""

_EXTRACT_CLAIMS_SYSTEM = """You extract atomic factual claims from a playlist explanation.

Rules:
- Each claim is one short, self-contained factual statement (mood, genre, duration, artists, flow, language, exclusions).
- Skip vague praise with no factual content (e.g. "great playlist", "hope you enjoy").
- Do not merge unrelated facts into one claim.
- Preserve the language of the explanation.
{format_instructions}"""

_EXTRACT_CLAIMS_USER = """Generated explanation:
{explanation}
"""

_CLAIM_ATTRIBUTION_SYSTEM = """You judge whether each factual claim can be inferred from the track contexts ONLY.

Mark attributable=true only if the claim is directly supported by artist names, moods, genres,
descriptions, or durations present in the contexts — not from the user query alone.
Mark attributable=false for invented facts, wrong artists, contradictions, or unsupported generalizations.
{format_instructions}"""

_CLAIM_ATTRIBUTION_USER = """User request (background only; contexts are the source of truth):
{query}

Track contexts:
{contexts}

Claims to judge (one attributable true/false per claim_index):
{claims_block}
"""

_GENERATE_QUESTIONS_SYSTEM = """You generate paraphrased PLAYLIST REQUESTS (user-style queries) that the given answer satisfies.

Given ONLY the generated playlist response (tracks + explanation), write exactly {num_questions}
natural-language requests a user might have typed to obtain this playlist.

Each item MUST:
- Be a playlist recommendation request (mood, activity, genre, duration intent, language, exclusions).
- Reflect the substantive music intent evidenced in the response (not metadata about the response itself).
- Be phrased as a user query, e.g. "I need calm instrumental music for studying, about 30 minutes."

Each item MUST NOT:
- Ask about the playlist as an artifact ("how many tracks", "what is the duration of the playlist",
  "list the song titles", "what is track 3").
- Be a yes/no question about whether something is in the playlist.
- Mention "playlist", "tracks included", or "explanation" as objects to inspect.

Vary wording (synonyms, different duration phrasing) while preserving the same underlying intent.
Use the same language as the explanation when possible.
{format_instructions}"""

_GENERATE_QUESTIONS_USER = """Generated playlist response:
{answer_block}
"""

_STATEMENT_RECALL_SYSTEM = """You evaluate RAG context recall for music playlist retrieval.

You receive a user request, a numbered list of reference requirements, and retrieved track contexts.
For each requirement, decide if the retrieved contexts COLLECTIVELY provide enough evidence that
a playlist built only from these tracks could satisfy that requirement.

Mark covered=true when at least several tracks in the context clearly support the requirement.
Mark covered=false when the context lacks that information or contradicts it.
Be strict on explicit exclusions (e.g. "no reggaeton" → covered only if contexts avoid reggaeton).

Return one judgment per statement_index (0-based, matching the numbered list).
{format_instructions}"""

_STATEMENT_RECALL_USER = """User request:
{query}

Reference requirements:
{statements_block}

Retrieved track contexts:
{contexts}
"""


def _relevance_chain():
    parser = PydanticOutputParser(pydantic_object=TrackRelevanceBatch)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _RELEVANCE_SYSTEM), ("user", _RELEVANCE_USER)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | _llm | parser


def _generate_questions_chain(num_questions: int):
    parser = PydanticOutputParser(pydantic_object=GeneratedQuestions)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _GENERATE_QUESTIONS_SYSTEM), ("user", _GENERATE_QUESTIONS_USER)]
    ).partial(
        format_instructions=parser.get_format_instructions(),
        num_questions=num_questions,
    )
    return prompt | _llm | parser


def _statement_recall_chain():
    parser = PydanticOutputParser(pydantic_object=StatementCoverageBatch)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _STATEMENT_RECALL_SYSTEM), ("user", _STATEMENT_RECALL_USER)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | _llm | parser


def _extract_claims_chain():
    parser = PydanticOutputParser(pydantic_object=ExtractedClaims)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _EXTRACT_CLAIMS_SYSTEM), ("user", _EXTRACT_CLAIMS_USER)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | _llm | parser


def _claim_attribution_chain():
    parser = PydanticOutputParser(pydantic_object=ClaimAttributionBatch)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _CLAIM_ATTRIBUTION_SYSTEM), ("user", _CLAIM_ATTRIBUTION_USER)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | _llm | parser


def extract_claims_from_explanation(explanation: str) -> list[str]:
    chain = _extract_claims_chain()
    result: ExtractedClaims = chain.invoke({"explanation": explanation})
    return [c.strip() for c in result.claims if c.strip()]


def judge_claim_attribution_batch(
    query: str,
    contexts: list[RetrievedTrack | PlaylistTrack],
    claims: list[str],
) -> ClaimAttributionBatch:
    if not claims:
        return ClaimAttributionBatch(judgments=[])
    claims_block = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    chain = _claim_attribution_chain()
    return chain.invoke(
        {
            "query": query,
            "contexts": format_context_block(contexts),
            "claims_block": claims_block,
        }
    )


def judge_faithfulness(
    query: str,
    contexts: list[RetrievedTrack | PlaylistTrack],
    explanation: str,
) -> FaithfulnessResult:
    
    claims = extract_claims_from_explanation(explanation)
    if not claims:
        return FaithfulnessResult(
            score=1.0,
            total_claims=0,
            supported_claims=0,
            claims=[],
            reasoning="No factual claims extracted from explanation.",
        )

    batch = judge_claim_attribution_batch(query, contexts, claims)
    attributable_by_index: dict[int, bool] = {}
    for j in batch.judgments:
        if 0 <= j.claim_index < len(claims):
            attributable_by_index[j.claim_index] = j.attributable

    supported = sum(
        1 for i in range(len(claims)) if attributable_by_index.get(i, False)
    )
    total = len(claims)
    score = supported / total
    return FaithfulnessResult(
        score=score,
        total_claims=total,
        supported_claims=supported,
        claims=claims,
        reasoning=(
            f"{supported}/{total} claims attributable to track contexts."
        ),
    )


def judge_statement_coverage_batch(
    query: str,
    statements: list[str],
    tracks: list[RetrievedTrack],
) -> StatementCoverageBatch:
    if not statements:
        return StatementCoverageBatch(judgments=[])
    statements_block = "\n".join(
        f"{i}. {s}" for i, s in enumerate(statements)
    )
    chain = _statement_recall_chain()
    return chain.invoke(
        {
            "query": query,
            "statements_block": statements_block,
            "contexts": format_context_block(tracks),
        }
    )


def judge_track_relevance_batch(
    query: str, tracks: list[RetrievedTrack]
) -> TrackRelevanceBatch:
    if not tracks:
        return TrackRelevanceBatch(judgments=[])
    block = "\n".join(
        f"- {format_track_context(t)}" for t in tracks
    )
    chain = _relevance_chain()
    return chain.invoke({"query": query, "tracks_block": block})


def generate_questions_from_answer(
    answer_block: str,
    num_questions: int | None = None,
) -> list[str]:
    n = num_questions or settings.eval_answer_relevance_num_questions
    chain = _generate_questions_chain(n)
    result: GeneratedQuestions = chain.invoke({"answer_block": answer_block})
    questions = [q.strip() for q in result.questions if q.strip()]
    return questions[:n]


def judge_answer_relevance(
    original_query: str, answer_block: str
) -> AnswerRelevanceResult:
    
    n = settings.eval_answer_relevance_num_questions
    questions = generate_questions_from_answer(answer_block, n)
    if not questions:
        return AnswerRelevanceResult(
            score=0.0,
            num_questions=0,
            reasoning="No questions generated from answer.",
        )

    e_original = embed_text(original_query)
    e_generated = embed_batch(questions)
    similarities = [_cosine_similarity(e_original, e_g) for e_g in e_generated]
    score = sum(similarities) / len(similarities)
    score = max(0.0, min(1.0, score))
    sims_str = ", ".join(f"{s:.3f}" for s in similarities)
    return AnswerRelevanceResult(
        score=score,
        num_questions=len(questions),
        generated_questions=questions,
        cosine_similarities=similarities,
        reasoning=(
            f"RAGAS: mean cos(E_g, E_o) over {len(questions)} questions "
            f"= {score:.3f} (per-question: {sims_str})."
        ),
    )
