"""Input/output guardrails -- Arize AX's risk-reduction/compliance story.

Both actually change the agent's behavior when triggered, not just log:
check_input can block a request before it ever reaches the LLM; check_output
can rewrite a non-compliant reply using the real facts the agent already
fetched via its tools. Each opens its own GUARDRAIL-kind span.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import config
import tracing

INPUT_DENY_RE = re.compile(
    r"\b(sue|lawsuit|bankrupt(?:cy)?|legal advice|press charges)\b", re.IGNORECASE
)
OUTPUT_GUARANTEE_RE = re.compile(
    r"\b(guarantee|100%|promise|definitely will|always works)\b", re.IGNORECASE
)
SCORE_RE = re.compile(r"\b(\d{3})\b")


@dataclass
class GuardrailResult:
    triggered: bool
    reason: str
    safe_response: str | None = None


def _parse_result(result: str) -> Any:
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {}


def _actual_score(tool_calls: list[dict]) -> int | None:
    for tc in tool_calls:
        if tc["name"] == "get_credit_report":
            data = _parse_result(tc["result"])
            if isinstance(data, dict) and "credit_score" in data:
                return data["credit_score"]
    return None


def _compliant_summary(tool_calls: list[dict]) -> str:
    facts = []
    for tc in tool_calls:
        data = _parse_result(tc["result"])
        if not isinstance(data, dict):
            continue
        if tc["name"] == "get_credit_report" and "credit_score" in data:
            facts.append(f"your credit score is {data['credit_score']} ({data.get('score_band', '')})")
        elif tc["name"] == "list_disputable_items":
            items = data.get("disputable_items", [])
            facts.append(
                f"disputable items on file: {', '.join(i['item_id'] for i in items)}"
                if items
                else "no disputable items on file"
            )
        elif tc["name"] == "check_dispute_eligibility":
            facts.append(
                f"{data.get('item_id')}: eligible={data.get('eligible')} ({data.get('reason', '')})"
            )

    hedge = "I can't guarantee a specific outcome -- disputes are reviewed case by case."
    if facts:
        return f"{hedge} Here's what I can confirm from your file: {'; '.join(facts)}."
    return hedge


def check_input(user_id: str, message: str) -> GuardrailResult:
    with tracing.traced_span(
        "input_guardrail",
        tracing.GUARDRAIL,
        input_value=message,
        attributes={"guardrail.name": "input_scope_guardrail", "user_id": user_id},
    ) as span:
        match = INPUT_DENY_RE.search(message)
        triggered = bool(match)
        reason = f"out-of-scope legal/financial-advice request (matched {match.group(0)!r})" if match else ""
        safe_response = None
        if triggered:
            safe_response = (
                f"That's outside what {config.agent_name()} can help with -- for legal or "
                "financial-advice questions like this, please talk to a licensed attorney or "
                f"financial advisor. I can help with your credit report, disputes, or general "
                f"{config.company_name()} questions instead."
            )
        span.set_attribute("guardrail.triggered", triggered)
        tracing.set_output(span, {"triggered": triggered, "reason": reason})
        return GuardrailResult(triggered=triggered, reason=reason, safe_response=safe_response)


def check_output(response_text: str, tool_calls: list[dict]) -> GuardrailResult:
    with tracing.traced_span(
        "output_guardrail",
        tracing.GUARDRAIL,
        input_value=response_text,
        attributes={"guardrail.name": "output_compliance_guardrail"},
    ) as span:
        reasons = []
        if OUTPUT_GUARANTEE_RE.search(response_text):
            reasons.append("guarantee/outcome-promise language")

        actual_score = _actual_score(tool_calls)
        if actual_score is not None:
            for match in SCORE_RE.findall(response_text):
                n = int(match)
                if 300 <= n <= 850 and n != actual_score:
                    reasons.append(f"stated score {n} does not match tool result ({actual_score})")
                    break

        triggered = bool(reasons)
        safe_response = _compliant_summary(tool_calls) if triggered else None
        span.set_attribute("guardrail.triggered", triggered)
        tracing.set_output(span, {"triggered": triggered, "reason": "; ".join(reasons)})
        return GuardrailResult(triggered=triggered, reason="; ".join(reasons), safe_response=safe_response)
