"""LLM-as-judge evaluator for the boolean is_valid_claim field."""

from __future__ import annotations

import json
from typing import Any

from config import openai_api_key, openai_model
from instrumentation import set_output, traced_span
from schema import ClaimJudgeResult

from .util import as_bool, as_output_dict, evaluation_result

JUDGE_SYSTEM = """You are scoring a document-extraction pipeline.

You will see:
1. The source deduction-notice text (including footers / disclaimers).
2. The extracted is_valid_claim boolean and claim_reason.

A valid claim is a real, processable vendor deduction. It is NOT valid if the
notice is a draft, placeholder, void, goodwill credit, or says do not process.

Decide:
- is_valid_per_document: would a deductions analyst treat this as a valid claim?
- extraction_correct: does extracted is_valid_claim match that judgment?
- explanation: 1-3 sentences citing the document.

Read the entire document, including small-print footers.
"""


def claim_validity_judge(output: Any, dataset_row: dict[str, Any] | None = None, **_: Any):
    """LLM-as-judge — is the extracted is_valid_claim justified by the source text?

    Scored against the ground-truth label on the dataset row. This is the
    subjective boolean-field evaluator; retailer and amount use code evals.
    """
    data = as_output_dict(output)
    row = dataset_row or {}
    source_text = data.get("source_text") or row.get("source_text") or ""
    extracted = as_bool(data.get("is_valid_claim"))
    expected = as_bool(row.get("expected_is_valid_claim"))
    reason = data.get("claim_reason") or ""

    with traced_span(
        "claim_validity_judge",
        "EVALUATOR",
        input_value=json.dumps(
            {
                "extracted_is_valid_claim": extracted,
                "expected_is_valid_claim": expected,
                "claim_reason": reason,
            }
        ),
    ) as span:
        if openai_api_key():
            judge = _judge_openai(source_text, extracted, reason)
        else:
            judge = _judge_mock(source_text, extracted)

        # Score against ground truth: extraction must match GT, and the judge
        # must confirm the extraction is consistent with the document.
        matches_gt = extracted is not None and extracted == expected
        extraction_ok = bool(judge.extraction_correct)
        passed = matches_gt and extraction_ok
        explanation = (
            f"Ground truth is_valid_claim={expected}. "
            f"Extracted is_valid_claim={extracted} "
            f"({'matches' if matches_gt else 'DIFFERS FROM'} GT). "
            f"Judge: valid_per_document={judge.is_valid_per_document}, "
            f"extraction_correct={judge.extraction_correct}. "
            f"{judge.explanation}"
        )
        result = evaluation_result(
            score=1.0 if passed else 0.0,
            label="pass" if passed else "fail",
            explanation=explanation,
        )
        set_output(
            span,
            {
                "label": "pass" if passed else "fail",
                "score": 1.0 if passed else 0.0,
                "judge": judge.model_dump(),
            },
        )
        return result


def _judge_openai(source_text: str, extracted: bool | None, reason: str) -> ClaimJudgeResult:
    from openai import OpenAI

    client = OpenAI()
    model = openai_model()
    user = (
        f"SOURCE DOCUMENT:\n{source_text}\n\n"
        f"EXTRACTED is_valid_claim: {extracted}\n"
        f"EXTRACTED claim_reason: {reason}\n"
    )
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=ClaimJudgeResult,
        temperature=0,
    )
    parse = getattr(client.chat.completions, "parse", None)
    if parse is None:
        completion = client.beta.chat.completions.parse(**kwargs)
    else:
        completion = parse(**kwargs)
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("LLM judge returned no structured result")
    return parsed


def _judge_mock(source_text: str, extracted: bool | None) -> ClaimJudgeResult:
    """No-key fallback: still emit an EVALUATOR span with the same result shape."""
    lowered = source_text.lower()
    invalid = any(
        marker in lowered
        for marker in (
            "internal draft",
            "do not process",
            "not a valid deduction",
            "goodwill",
            "not a contractual",
            "placeholder",
            "negative deduction",
        )
    )
    valid_per_doc = not invalid
    extraction_correct = extracted is not None and extracted == valid_per_doc
    explanation = (
        "Mock judge (no OPENAI_API_KEY): flagged invalid because of draft/void/"
        "goodwill language in the source text."
        if invalid
        else "Mock judge (no OPENAI_API_KEY): no void/draft/goodwill markers; treating as valid."
    )
    return ClaimJudgeResult(
        is_valid_per_document=valid_per_doc,
        extraction_correct=extraction_correct,
        explanation=explanation,
    )
