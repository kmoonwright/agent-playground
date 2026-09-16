"""The 4 evaluator types: code, LLM-judge, agent/holistic, remote-agent stub.

Each follows the Arize experiment convention:

    def evaluator(output, dataset_row) -> EvaluationResult

`output` is the dict returned by run_experiment.task() (an AgentResult,
dataclasses.asdict'd). `dataset_row` is one row from dataset_gen.py.
"""

from __future__ import annotations

import json
import time
from typing import Any

import config
import tracing


def _result(score: float, label: str, explanation: str):
    from arize.experiments import EvaluationResult

    return EvaluationResult(score=score, label=label, explanation=explanation)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


# ---------------------------------------------------------------------------
# 1. Code eval -- deterministic, no LLM. "Correct tool called" + "tool called
#    correctly" (name AND args match).
# ---------------------------------------------------------------------------
def code_eval(output: Any, dataset_row: dict | None = None, **_: Any):
    row = dataset_row or {}
    output = _as_dict(output)
    tool_calls = output.get("tool_calls") or []
    expected_tool = row.get("expected_tool")
    expected_args = row.get("expected_tool_args") or {}

    if expected_tool is None:
        called_any = bool(tool_calls)
        passed = not called_any
        explanation = (
            "No tool expected; agent correctly answered without one."
            if passed
            else f"No tool expected, but agent called: {[tc['name'] for tc in tool_calls]}."
        )
        return _result(1.0 if passed else 0.0, "pass" if passed else "fail", explanation)

    match = next((tc for tc in tool_calls if tc.get("name") == expected_tool), None)
    if match is None:
        return _result(
            0.0,
            "fail",
            f"Expected tool '{expected_tool}' was never called. "
            f"Called: {[tc['name'] for tc in tool_calls]}.",
        )

    args_ok = all(match.get("args", {}).get(k) == v for k, v in expected_args.items())
    passed = args_ok
    explanation = (
        f"Tool '{expected_tool}' called with args {match.get('args')}, expected {expected_args}."
    )
    return _result(1.0 if passed else 0.0, "pass" if passed else "fail", explanation)


# ---------------------------------------------------------------------------
# 2. LLM eval -- Claude/GPT-as-judge, structured multi-axis JSON, not a
#    single pass/fail.
# ---------------------------------------------------------------------------
_LLM_JUDGE_PROMPT = """You are grading a customer support reply from a credit-coach assistant.

User asked: {message}
Assistant replied: {response}

Score three axes, each "yes", "partial", or "no":
- correctness: is the reply factually consistent with what a credit coach should say (no fabricated numbers/claims)?
- completeness: does the reply actually address the user's question?
- tone: is the reply plain-English, empathetic, and appropriate for a fintech support context?

Respond with ONLY a JSON object: {{"correctness": "...", "completeness": "...", "tone": "...", "explanation": "..."}}
"""

_AXIS_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def _judge(prompt: str) -> dict:
    if not config.openai_api_key():
        return {
            "correctness": "partial",
            "completeness": "partial",
            "tone": "yes",
            "explanation": "Mock judge (no OPENAI_API_KEY set) -- offline placeholder score.",
        }
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = completion.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"correctness": "no", "completeness": "no", "tone": "no", "explanation": text}


def llm_eval(output: Any, dataset_row: dict | None = None, **_: Any):
    row = dataset_row or {}
    output = _as_dict(output)
    response_text = output.get("response_text", "")
    message = row.get("message", "")

    with tracing.traced_span(
        "llm_eval", tracing.EVALUATOR, input_value=json.dumps({"message": message, "response": response_text})
    ) as span:
        judged = _judge(_LLM_JUDGE_PROMPT.format(message=message, response=response_text))
        axes = ["correctness", "completeness", "tone"]
        scores = [_AXIS_SCORE.get(str(judged.get(a, "no")).lower(), 0.0) for a in axes]
        score = sum(scores) / len(scores)
        label = "pass" if score >= 0.75 else "partial" if score >= 0.25 else "fail"
        explanation = (
            f"correctness={judged.get('correctness')}, completeness={judged.get('completeness')}, "
            f"tone={judged.get('tone')}. {judged.get('explanation', '')}"
        )
        result = _result(score, label, explanation)
        tracing.set_output(span, {"score": score, "label": label, "judged": judged})
        return result


# ---------------------------------------------------------------------------
# 3. Agent eval -- holistic: judges the *entire* trace (tool calls + final
#    answer) against expected_facts, not just the isolated tool call.
# ---------------------------------------------------------------------------
_AGENT_JUDGE_PROMPT = """You are checking whether a credit-coach agent's full behavior matches expected facts.

User asked: {message}
Agent's tool calls: {tool_calls}
Agent's final reply: {response}
Expected facts (ground truth): {expected_facts}

Does the agent's reply correctly surface/reflect the expected facts (e.g. the
right credit score, the right eligibility answer, the right CTA)? This is a
holistic check -- a correct tool call with a reply that misstates the result
should still fail.

Respond with ONLY a JSON object: {{"matches_expected_facts": "yes"|"partial"|"no", "explanation": "..."}}
"""


def _agent_judge(output: dict, row: dict) -> dict:
    prompt = _AGENT_JUDGE_PROMPT.format(
        message=row.get("message", ""),
        tool_calls=json.dumps(output.get("tool_calls") or []),
        response=output.get("response_text", ""),
        expected_facts=json.dumps(row.get("expected_facts") or {}),
    )
    if not config.openai_api_key():
        return {
            "matches_expected_facts": "partial",
            "explanation": "Mock judge (no OPENAI_API_KEY set) -- offline placeholder score.",
        }
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = completion.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"matches_expected_facts": "no", "explanation": text}


def agent_eval(output: Any, dataset_row: dict | None = None, **_: Any):
    row = dataset_row or {}
    output = _as_dict(output)

    with tracing.traced_span(
        "agent_eval",
        tracing.EVALUATOR,
        input_value=json.dumps({"expected_facts": row.get("expected_facts")}),
    ) as span:
        judged = _agent_judge(output, row)
        verdict = str(judged.get("matches_expected_facts", "no")).lower()
        score = _AXIS_SCORE.get(verdict, 0.0)
        label = "pass" if score == 1.0 else "partial" if score == 0.5 else "fail"
        result = _result(score, label, judged.get("explanation", ""))
        tracing.set_output(span, {"score": score, "label": label, "judged": judged})
        return result


# ---------------------------------------------------------------------------
# 4. Remote eval -- stub for "call out to the customer's own external judge."
#    Reuses the same holistic judge logic as agent_eval (still genuinely
#    agent/LLM-based, not a rule-based duplicate of code_eval) but wrapped to
#    simulate an external round-trip: its own span, a small fixed delay.
# ---------------------------------------------------------------------------
def remote_eval(output: Any, dataset_row: dict | None = None, **_: Any):
    row = dataset_row or {}
    output = _as_dict(output)

    with tracing.traced_span(
        "remote_eval_stub",
        tracing.EVALUATOR,
        input_value=json.dumps({"expected_facts": row.get("expected_facts")}),
        attributes={"remote.simulated": True},
    ) as span:
        time.sleep(0.2)  # simulate a network round-trip to an external judge
        judged = _agent_judge(output, row)
        verdict = str(judged.get("matches_expected_facts", "no")).lower()
        score = _AXIS_SCORE.get(verdict, 0.0)
        label = "pass" if score == 1.0 else "partial" if score == 0.5 else "fail"
        explanation = f"[simulated remote judge] {judged.get('explanation', '')}"
        result = _result(score, label, explanation)
        tracing.set_output(span, {"score": score, "label": label, "judged": judged})
        return result


EVALUATORS = [code_eval, llm_eval, agent_eval, remote_eval]
