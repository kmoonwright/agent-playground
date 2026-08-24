"""The two tools the agent is allowed to call.

read_file  — look at the real file so line numbers are not guesses from the diff.
submit_review — the only way a review is recorded. Chat prose is discarded.

Nothing here talks to GitHub. submit_review writes to out/.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from schema import Review

READ_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a file at PR head. Use this to verify line numbers and inspect "
            "surrounding context. Only files listed in the Files Changed manifest "
            "are readable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path, exactly as listed in Files Changed",
                }
            },
            "required": ["path"],
        },
    },
}

SUBMIT_REVIEW_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": (
            "Submit the finished code review. Call this exactly once, at the end. "
            "It records the review locally — it does NOT post to GitHub. "
            "Only set `line` on a finding after you verified it with read_file. "
            "An empty findings list is correct for a genuinely clean PR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "taste_rating": {
                    "type": "string",
                    "enum": ["good_taste", "acceptable", "needs_improvement"],
                },
                "verdict": {
                    "type": "string",
                    "enum": ["worth_merging", "needs_rework"],
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "key_insight": {
                    "type": "string",
                    "description": "One sentence: the most important observation",
                },
                "findings": {
                    "type": "array",
                    "description": "Structured findings. Empty list is correct for a clean PR.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "line": {"type": ["integer", "null"]},
                            "severity": {
                                "type": "string",
                                "enum": [
                                    "critical",
                                    "improvement",
                                    "style",
                                    "testing",
                                ],
                            },
                            "category": {"type": "string"},
                            "comment": {"type": "string"},
                        },
                        "required": ["file", "severity", "category", "comment"],
                    },
                },
                "review_markdown": {
                    "type": "string",
                    "description": "Full human-readable review a maintainer would read",
                },
            },
            "required": [
                "taste_rating",
                "verdict",
                "risk_level",
                "key_insight",
                "findings",
                "review_markdown",
            ],
        },
    },
}

TOOLS = [READ_FILE_TOOL, SUBMIT_REVIEW_TOOL]


def parse_arguments(raw: str) -> dict[str, Any]:
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool arguments were not valid JSON: {exc}") from exc
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return args


def save_review(arguments: dict[str, Any], artifact_path: Path, pr_slug: str) -> tuple[Review, str]:
    """Validate, write JSON + markdown artifacts, return (review, observation)."""
    try:
        review = Review.model_validate(arguments)
    except ValidationError as exc:
        raise ValueError(f"submit_review payload failed validation:\n{exc}") from exc

    record = {
        "pr": pr_slug,
        "submitted_at": datetime.now(UTC).isoformat(),
        **review.model_dump(mode="json"),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(record, indent=2))
    artifact_path.with_suffix(".md").write_text(review.review_markdown)

    unverified = sum(1 for f in review.findings if f.line is None)
    note = (
        f"{unverified} finding(s) submitted without a line number."
        if unverified
        else "All findings carry a line number."
        if review.findings
        else "No findings (clean PR)."
    )
    observation = (
        f"Review recorded ({len(review.findings)} findings) at {artifact_path}. "
        f"{note} Nothing was posted to GitHub. You are done — do not call this tool again."
    )
    return review, observation
