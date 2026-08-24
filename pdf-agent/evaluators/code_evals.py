"""Deterministic code evaluators for string (retailer) and int (amount) fields."""

from __future__ import annotations

from typing import Any

from config import AMOUNT_MAX_CENTS, AMOUNT_MIN_CENTS, KNOWN_RETAILERS

from .util import as_output_dict, evaluation_result


def retailer_name_match(output: Any, dataset_row: dict[str, Any] | None = None, **_: Any):
    """Code eval — extracted retailer_name must be in the known enum.

    Case-insensitive exact match after stripping common suffixes (Inc, LLC).
    Unknown retailers (e.g. Kroger) fail. This is the string-typed field check.
    """
    data = as_output_dict(output)
    name = data.get("retailer_name")
    if not isinstance(name, str) or not name.strip():
        return evaluation_result(
            score=0.0,
            label="fail",
            explanation=f"retailer_name is missing or not a string (got {name!r}).",
        )

    normalized = _normalize_retailer(name)
    known = {_normalize_retailer(r): r for r in KNOWN_RETAILERS}
    if normalized in known:
        expected = None
        if dataset_row:
            expected = dataset_row.get("expected_retailer_name")
        extra = ""
        if expected and _normalize_retailer(str(expected)) != normalized:
            extra = (
                f" Name is in the enum but does not match ground truth "
                f"({expected!r}). Still passing the enum check."
            )
        return evaluation_result(
            score=1.0,
            label="pass",
            explanation=f"{name!r} matches known retailer {known[normalized]!r}.{extra}",
        )

    return evaluation_result(
        score=0.0,
        label="fail",
        explanation=(
            f"{name!r} is not in the known retailer list "
            f"{list(KNOWN_RETAILERS)}. Flagged as out-of-enum."
        ),
    )


def deduction_amount_valid(output: Any, dataset_row: dict[str, Any] | None = None, **_: Any):
    """Code eval — deduction_amount must be an int in [0, $100,000] (cents)."""
    data = as_output_dict(output)
    amount = data.get("deduction_amount")

    if isinstance(amount, bool) or not isinstance(amount, int):
        return evaluation_result(
            score=0.0,
            label="fail",
            explanation=(
                f"deduction_amount failed the int type check "
                f"(got {amount!r}, type={type(amount).__name__})."
            ),
        )

    if amount < AMOUNT_MIN_CENTS:
        return evaluation_result(
            score=0.0,
            label="fail",
            explanation=(
                f"deduction_amount {amount} cents is negative. "
                f"Valid range is [{AMOUNT_MIN_CENTS}, {AMOUNT_MAX_CENTS}] cents."
            ),
        )
    if amount > AMOUNT_MAX_CENTS:
        dollars = amount / 100
        cap = AMOUNT_MAX_CENTS / 100
        return evaluation_result(
            score=0.0,
            label="fail",
            explanation=(
                f"deduction_amount {amount} cents (${dollars:,.2f}) exceeds the "
                f"sanity cap of {AMOUNT_MAX_CENTS} cents (${cap:,.2f})."
            ),
        )

    dollars = amount / 100
    return evaluation_result(
        score=1.0,
        label="pass",
        explanation=(
            f"deduction_amount {amount} cents (${dollars:,.2f}) is an int inside "
            f"[{AMOUNT_MIN_CENTS}, {AMOUNT_MAX_CENTS}]."
        ),
    )


def _normalize_retailer(name: str) -> str:
    cleaned = name.strip()
    for suffix in (" Inc.", " Inc", " LLC", " Ltd.", " Ltd"):
        if cleaned.lower().endswith(suffix.lower()):
            cleaned = cleaned[: -len(suffix)]
    return cleaned.strip().lower()
