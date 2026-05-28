import json
from pathlib import Path

from playlist_rag.eval.schemas import EvalCase


def load_eval_cases(path: Path) -> list[EvalCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"eval dataset not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        cases = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(EvalCase.model_validate(json.loads(line)))
        return cases

    data = json.loads(text)
    if isinstance(data, list):
        return [EvalCase.model_validate(item) for item in data]
    if isinstance(data, dict) and "cases" in data:
        return [EvalCase.model_validate(item) for item in data["cases"]]
    raise ValueError(f"unsupported eval dataset format: {path}")
