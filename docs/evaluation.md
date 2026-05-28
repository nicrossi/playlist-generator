# Evaluation (`playlist_rag.eval`)

RAG-style evaluation for the playlist generation pipeline. Measures retrieval
quality and generation quality using LLM-as-judge metrics adapted to music
retrieval (tracks as contexts, playlist + explanation as the answer).

## Metrics

| Metric | Type | Definition in this project |
|---|---|---|
| **Context precision** | Retrieval | RAGAS Context Precision@K: rank-weighted precision over retrieved tracks (LLM relevance per rank). |
| **Context recall** | Retrieval (RAGAS-style) | Fraction of `reference_statements` in the eval case that the retrieved track contexts collectively support (LLM judge). |
| **Faithfulness** | Generation | RAGAS: fraction of factual claims in the explanation attributable to playlist track contexts. |
| **Answer relevance** | Generation | RAGAS answer relevancy: mean cosine similarity between embeddings of questions generated from the answer and the original user query. |

Contexts are track records, not text chunks; the answer is a structured playlist.

## Formulas

**Context precision** (RAGAS Context Precision@K, up to `eval_max_retrieved_to_judge` tracks in retrieval order):

Let \(v_k \in \{0,1\}\) be LLM-judged relevance at rank \(k\) (1 = most similar from `hybrid_search`).

\[
\text{Precision@}k = \frac{\text{true positives@}k}{k}
= \frac{\sum_{i=1}^{k} v_i}{k}
\]

\[
\text{Context Precision@}K = \frac{\sum_{k=1}^{K} (\text{Precision@}k \cdot v_k)}{\sum_{k=1}^{K} v_k}
\]

Relevant items ranked higher contribute more than the same count placed low in the list.
If no track is relevant, the score is \(0\).

**Context recall** (RAGAS-style, requires `reference_statements`):

\[
\text{context\_recall} = \frac{\#\{\text{statements with covered=true}\}}{|\text{reference\_statements}|}
\]

One LLM call judges all statements against the same retrieved context block.

**Faithfulness** (RAGAS, over `explanation` vs playlist track contexts):

\[
\text{faithfulness} = \frac{|\{\text{claims in answer attributable to context}\}|}{|\{\text{claims in answer}\}|}
\]

1. LLM extracts atomic factual claims from the generated explanation.
2. LLM marks each claim as attributable (or not) from track metadata/descriptions only.
3. If no claims are extracted, score is \(1.0\) (vacuous).

**Answer relevance** (RAGAS answer relevancy):

\[
\text{answer relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_{g_i}, E_o)
\]

- \(E_o\): embedding of the **original user query** (not the full answer block).
- \(E_{g_i}\): embedding of the \(i\)-th question **generated from the answer only** (playlist + explanation).
- \(N\): `eval_answer_relevance_num_questions` (default 3).

1. LLM generates \(N\) **paraphrased playlist requests** (user-style queries) implied by the
   answer — not meta-questions about track count or playlist structure.
2. Embed original user query and each generated request (`text-embedding-3-small`).
3. Average cosine similarities.

## Prerequisites

- Indexed catalog (`python -m playlist_rag.cli.index`)
- HNSW migration (`alembic upgrade head`)
- `OPENAI_API_KEY` in `.env` (generation + judges)

## Dataset format

`eval/queries.jsonl` — one JSON object per line:

```json
{
  "id": "calm_study_30m",
  "query": "calm instrumental music for studying 30 minutes",
  "reference_statements": [
    "Tracks should feel calm and low-energy, suitable for concentration.",
    "Selection should favor instrumental or low-vocal content for studying."
  ],
  "notes": "optional"
}
```

- **`reference_statements`**: rubric requirements for **context recall**.
  Leave `[]` to skip that metric for the case.

## Quickstart

```bash
# Full eval
python -m playlist_rag.cli.evaluate --dataset eval/queries.jsonl

# Cheaper smoke run (1 query, skip retrieval LLM judges)
python -m playlist_rag.cli.evaluate --dataset eval/queries.jsonl --limit 1 \
  --skip-retrieval-judge --no-llm-explain

# Save JSON report
python -m playlist_rag.cli.evaluate --dataset eval/queries.jsonl \
  --output reports/eval_report.json --include-runs
```

## CLI flags

| Flag | Purpose |
|---|---|
| `--dataset PATH` | JSON or JSONL eval cases (default `eval/queries.jsonl`) |
| `--output PATH` | Write full `EvalReport` as JSON |
| `--limit N` | Evaluate only first N cases |
| `--top-k N` | Retrieval pool size passed to generation |
| `--no-llm-explain` | Use template explanation during generation (cheaper) |
| `--skip-retrieval-judge` | Skip context precision + context recall (LLM judges) |
| `--skip-generation-judge` | Skip faithfulness + answer relevance |
| `--include-runs` | Embed full `GenerationRun` in output JSON |
| `-v` | DEBUG logging |

## Module layout

```
playlist_rag/eval/
├── schemas.py      EvalCase, QueryMetrics, EvalReport
├── dataset.py      load JSON / JSONL
├── context.py      format tracks for judges
├── judge.py        LLM judges (track relevance, statement recall, faithfulness, answer relevance)
├── metrics.py      compute_query_metrics()
└── runner.py       run_evaluation()
```

## Configuration

| Setting | Default | Notes |
|---|---|---|
| `eval_judge_batch_size` | 10 | Tracks per relevance judge call |
| `eval_max_retrieved_to_judge` | 40 | Cap on tracks used for precision + RAGAS recall |
| `eval_judge_temperature` | 0.0 | Judge LLM temperature |
| `eval_answer_relevance_num_questions` | 3 | Questions generated per answer for answer relevancy |

## Interpreting results

- **Low context precision**: retrieval or embeddings mismatch the query.
- **Low context recall**: retrieved pool lacks information to satisfy rubric
  requirements (even if individual tracks look relevant).
- **Low faithfulness**: explanation invents facts vs track contexts.
- **Low answer relevance**: paraphrased requests derived from the answer are not
  similar to the original user query (irrelevant or incomplete response).

## Cost note

Per eval case (full judges), approximate Chat calls:

- Generation: parse + explain (1 embedding)
- Context precision: ~ceil(40 / batch_size)
- Context recall: **1** call per case
- Faithfulness: **2** calls (extract claims + attribute claims)
- Answer relevance: **1** LLM call + **N+1** embeddings (N questions + original query)

## Related docs

- [`generation_pipeline.md`](generation_pipeline.md)
- [`indexing_pipeline.md`](indexing_pipeline.md)
