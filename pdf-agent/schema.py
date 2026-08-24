"""Structured extraction schema — int, boolean, and string fields."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """Fields PDF extraction pipelines pull off a deduction notice."""

    retailer_name: str = Field(
        description="Retailer on the notice. Known set: Walmart, Target, CVS, Amazon."
    )
    deduction_amount: int = Field(
        description="Claimed deduction amount in US cents (integer). $125.00 → 12500."
    )
    is_valid_claim: bool = Field(
        description=(
            "True only if this is a real, processable deduction claim. "
            "False for drafts, placeholders, void notices, or goodwill credits."
        )
    )
    claim_reason: str = Field(
        description="Short free-text reason copied from the notice."
    )


class ParsedDocument(BaseModel):
    """Typed pipeline output after JSON parse + Pydantic validation."""

    retailer_name: str | None = None
    deduction_amount: int | None = None
    is_valid_claim: bool | None = None
    claim_reason: str | None = None
    parse_ok: bool = True
    parse_error: str | None = None

    def as_output(self) -> dict[str, Any]:
        return self.model_dump()


class ClaimJudgeResult(BaseModel):
    is_valid_per_document: bool = Field(
        description="Whether the source document describes a valid, processable claim."
    )
    extraction_correct: bool = Field(
        description="Whether the extracted is_valid_claim matches the document."
    )
    explanation: str
