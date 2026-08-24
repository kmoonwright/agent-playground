"""Structured review the agent must submit via the submit_review tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "improvement", "style", "testing"]
TasteRating = Literal["good_taste", "acceptable", "needs_improvement"]
Verdict = Literal["worth_merging", "needs_rework"]
RiskLevel = Literal["low", "medium", "high"]


class Finding(BaseModel):
    """One observation, tied to a location when the agent actually read the file."""

    file: str = Field(description="Repo-relative path the finding applies to")
    line: int | None = Field(
        default=None,
        description=(
            "Line number in the file at PR head. Only set this if you verified it "
            "by calling read_file — do not count lines from a diff hunk header."
        ),
    )
    severity: Severity
    category: str = Field(
        description="Short label, e.g. 'Security', 'Logic', 'Breaking Change'"
    )
    comment: str = Field(description="The actionable feedback, in full sentences")


class Review(BaseModel):
    taste_rating: TasteRating
    verdict: Verdict
    risk_level: RiskLevel
    key_insight: str = Field(
        description="One sentence: the most important observation about this PR"
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Structured findings. Empty list is correct for a clean PR.",
    )
    review_markdown: str = Field(
        description="Full human-readable review a maintainer would read"
    )
