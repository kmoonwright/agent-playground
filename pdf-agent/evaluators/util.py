"""Helpers shared by code and LLM evaluators."""

from __future__ import annotations

import json
from typing import Any


def as_output_dict(output: Any) -> dict[str, Any]:
    if output is None:
        return {}
    if isinstance(output, dict):
        return output
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if isinstance(output, str):
        try:
            loaded = json.loads(output)
        except json.JSONDecodeError:
            return {"raw": output}
        if isinstance(loaded, dict):
            return loaded
    return {"value": output}


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return None
    return bool(value)


def evaluation_result(*, score: float, label: str, explanation: str):
    try:
        from arize.experiments import EvaluationResult
    except ImportError:
        from arize.experiments.types import EvaluationResult  # type: ignore

    return EvaluationResult(score=score, label=label, explanation=explanation)
