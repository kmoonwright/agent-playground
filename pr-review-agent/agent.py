"""Visible tool-calling loop. This file is the agent.

Each iteration: call the model → run any tool calls → append observations.
submit_review ends the run. If the model writes a review as chat instead of
calling the tool, we nudge once. A second miss is a real failure.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import openai_model, require_openai
from instrumentation import (
    ensure_tracing,
    flush_tracing,
    set_output,
    traced_span,
    using_review_context,
)
from prompt import NUDGE, SYSTEM_PROMPT, build_prompt
from schema import Review
from source import ReviewTarget
from tools import TOOLS, parse_arguments, save_review


def run_review(
    target: ReviewTarget,
    *,
    artifact_path: Path,
    max_iterations: int,
    model: str | None = None,
) -> Review | None:
    require_openai()
    ensure_tracing()
    from openai import OpenAI

    client = OpenAI()
    resolved_model = model or openai_model()
    user_prompt = build_prompt(target)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    submitted: Review | None = None
    nudged = False

    extra = {
        "repo": target.repo,
        "pr_number": target.number,
        "pr_title": target.title,
        "model": resolved_model,
    }

    try:
        with using_review_context(target.slug, extra):
            with traced_span(
                "review_pull_request",
                "CHAIN",
                input_value=user_prompt,
                attributes={
                    "review.pr": target.slug,
                    "review.model": resolved_model,
                    "review.changed_files": len(target.files),
                },
            ) as root:
                for step in range(1, max_iterations + 1):
                    print(f"  iteration {step}/{max_iterations}")
                    response = client.chat.completions.create(
                        model=resolved_model,
                        messages=messages,
                        tools=TOOLS,
                    )
                    choice = response.choices[0].message
                    messages.append(_assistant_message(choice))

                    if not choice.tool_calls:
                        if not nudged:
                            print("  no tool call — nudging once")
                            nudged = True
                            messages.append({"role": "user", "content": NUDGE})
                            continue
                        break

                    stop = False
                    for call in choice.tool_calls:
                        name = call.function.name
                        raw_args = call.function.arguments
                        observation, maybe_review = _run_tool(
                            name,
                            raw_args,
                            target=target,
                            artifact_path=artifact_path,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": observation,
                            }
                        )
                        if maybe_review is not None:
                            submitted = maybe_review
                            stop = True
                            break
                    if stop:
                        break

                _record_chain_result(root, submitted)
    finally:
        flush_tracing()

    return submitted


def _run_tool(
    name: str,
    raw_args: str,
    *,
    target: ReviewTarget,
    artifact_path: Path,
) -> tuple[str, Review | None]:
    """Execute one tool call under a TOOL span. Returns (observation, review|None)."""
    with traced_span(
        name,
        "TOOL",
        input_value=raw_args,
        attributes={"tool.name": name},
    ) as span:
        try:
            args = parse_arguments(raw_args)
        except ValueError as exc:
            set_output(span, str(exc))
            return str(exc), None

        if name == "read_file":
            path = str(args.get("path", ""))
            print(f"    read_file {path}")
            result = target.read_file(path)
            set_output(span, result)
            return result, None

        if name == "submit_review":
            print("    submit_review")
            try:
                review, observation = save_review(args, artifact_path, target.slug)
            except ValueError as exc:
                set_output(span, str(exc))
                return str(exc), None
            set_output(span, review.model_dump(mode="json"))
            return observation, review

        unknown = f"error: unknown tool {name!r}. Use read_file or submit_review."
        set_output(span, unknown)
        return unknown, None


def _assistant_message(choice) -> dict:
    message: dict = {"role": "assistant", "content": choice.content or ""}
    if choice.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in choice.tool_calls
        ]
    return message


def _record_chain_result(span, submitted: Review | None) -> None:
    from opentelemetry.trace import Status, StatusCode

    if submitted is None:
        set_output(span, "")
        span.set_attribute("review.submitted", False)
        span.set_status(
            Status(StatusCode.ERROR, "agent finished without submitting a review")
        )
        return

    findings = submitted.findings
    set_output(span, submitted.review_markdown)
    span.set_attribute("review.submitted", True)
    span.set_attribute("review.taste_rating", submitted.taste_rating)
    span.set_attribute("review.verdict", submitted.verdict)
    span.set_attribute("review.risk_level", submitted.risk_level)
    span.set_attribute("review.key_insight", submitted.key_insight)
    span.set_attribute("review.finding_count", len(findings))
    span.set_attribute(
        "review.findings_without_line",
        sum(1 for f in findings if f.line is None),
    )
    span.set_attribute(
        "review.findings",
        json.dumps([f.model_dump(mode="json") for f in findings]),
    )
    span.set_status(Status(StatusCode.OK))
