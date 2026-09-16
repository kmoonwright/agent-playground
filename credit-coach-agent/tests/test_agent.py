"""Offline selftest: no network calls, no API keys needed.

Monkeypatches openai.OpenAI with a scripted fake client to exercise the
tool-calling loop, and swaps in an in-memory span exporter to assert the
credit-coach-agent-turn / tool span nesting that the whole tracing design
depends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import tools  # noqa: E402
import tracing  # noqa: E402


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = _FakeFunction(name, json.dumps(arguments))


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        data = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeCompletion:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    def __init__(self, messages):
        self._script = iter(messages)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeCompletion(next(self._script))


class _FakeChat:
    def __init__(self, messages):
        self.completions = _FakeCompletions(messages)


class _FakeOpenAI:
    def __init__(self, messages):
        self.chat = _FakeChat(messages)


def _patch_openai(monkeypatch, messages):
    import openai

    fake = _FakeOpenAI(messages)
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: fake)
    return fake


def test_single_tool_call_then_final_answer(monkeypatch):
    tool_call = _FakeToolCall("call_1", "get_credit_report", {"user_id": "P01"})
    _patch_openai(
        monkeypatch,
        [
            _FakeMessage(tool_calls=[tool_call]),
            _FakeMessage(content="Your score is 548."),
        ],
    )

    result = agent.run_agent("P01", "what's my score?")

    assert result.response_text == "Your score is 548."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "get_credit_report"
    assert result.tool_calls[0]["args"] == {"user_id": "P01"}
    assert "548" in result.tool_calls[0]["result"]


def test_multiple_tool_calls_in_one_turn_each_get_their_own_result(monkeypatch):
    calls = [
        _FakeToolCall("call_1", "get_credit_report", {"user_id": "P01"}),
        _FakeToolCall("call_2", "list_disputable_items", {"user_id": "P01"}),
    ]
    _patch_openai(
        monkeypatch,
        [
            _FakeMessage(tool_calls=calls),
            _FakeMessage(content="Here's both."),
        ],
    )

    result = agent.run_agent("P01", "tell me everything")

    assert [tc["name"] for tc in result.tool_calls] == ["get_credit_report", "list_disputable_items"]


def test_iteration_budget_is_respected(monkeypatch):
    import config

    never_ending = [
        _FakeMessage(tool_calls=[_FakeToolCall(f"call_{i}", "get_credit_report", {"user_id": "P01"})])
        for i in range(config.MAX_ITERATIONS + 2)
    ]
    _patch_openai(monkeypatch, never_ending)

    result = agent.run_agent("P01", "loop forever")

    assert len(result.tool_calls) == config.MAX_ITERATIONS
    assert result.response_text == ""


def test_unknown_tool_becomes_an_observation_not_an_exception():
    observation = tools.dispatch("not_a_real_tool", {})
    assert "unknown tool" in observation


def test_agent_turn_span_nests_the_tool_span(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "get_tracer", lambda: provider.get_tracer("test"))

    tool_call = _FakeToolCall("call_1", "get_credit_report", {"user_id": "P01"})
    _patch_openai(
        monkeypatch,
        [
            _FakeMessage(tool_calls=[tool_call]),
            _FakeMessage(content="Your score is 548."),
        ],
    )

    agent.run_agent("P01", "what's my score?")

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "credit-coach-agent-turn" in spans
    assert "get_credit_report" in spans
    chain_span = spans["credit-coach-agent-turn"]
    tool_span = spans["get_credit_report"]
    assert chain_span.attributes["openinference.span.kind"] == "CHAIN"
    assert tool_span.attributes["openinference.span.kind"] == "TOOL"
    assert tool_span.parent.span_id == chain_span.context.span_id


def test_using_session_tags_every_span_across_multiple_separate_calls(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "get_tracer", lambda: provider.get_tracer("test"))

    tool_call = _FakeToolCall("call_1", "get_credit_report", {"user_id": "P01"})

    _patch_openai(
        monkeypatch,
        [_FakeMessage(tool_calls=[tool_call]), _FakeMessage(content="turn one done.")],
    )
    with tracing.using_session("s1", extra={"user_id": "P01"}):
        agent.run_agent("P01", "what's my score?")

    _patch_openai(monkeypatch, [_FakeMessage(content="turn two done.")])
    with tracing.using_session("s1", extra={"user_id": "P01"}):
        agent.run_agent("P01", "anything else?")

    _patch_openai(monkeypatch, [_FakeMessage(content="a different session.")])
    with tracing.using_session("s2"):
        agent.run_agent("P01", "unrelated question")

    spans = exporter.get_finished_spans()
    chain_spans = [s for s in spans if s.name == "credit-coach-agent-turn"]
    tool_spans = [s for s in spans if s.name == "get_credit_report"]
    assert len(chain_spans) == 3
    assert len(tool_spans) == 1

    s1_chain_spans = [s for s in chain_spans if s.attributes.get("session.id") == "s1"]
    assert len(s1_chain_spans) == 2  # both turns of session s1
    assert tool_spans[0].attributes.get("session.id") == "s1"  # manual TOOL span too

    s2_chain_spans = [s for s in chain_spans if s.attributes.get("session.id") == "s2"]
    assert len(s2_chain_spans) == 1
    assert all(s.attributes.get("session.id") != "s1" for s in s2_chain_spans)


def test_input_guardrail_short_circuits_before_any_llm_call(monkeypatch):
    fake = _patch_openai(monkeypatch, [_FakeMessage(content="should never be reached")])

    result = agent.run_agent("P01", "Should I sue over this, or file for bankruptcy?")

    assert fake.chat.completions.calls == 0
    assert result.guardrails_triggered == ["input_scope_guardrail"]
    assert result.tool_calls == []
    assert result.response_text  # the guardrail's canned safe_response


def test_get_credit_report_nests_fake_mcp_and_db_spans(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "get_tracer", lambda: provider.get_tracer("test"))

    tool_call = _FakeToolCall("call_1", "get_credit_report", {"user_id": "P01"})
    _patch_openai(
        monkeypatch,
        [_FakeMessage(tool_calls=[tool_call]), _FakeMessage(content="Your score is 548.")],
    )

    agent.run_agent("P01", "what's my score?")

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "mcp.get_credit_report" in spans
    assert "db.query:credit_reports" in spans
    tool_span = spans["get_credit_report"]
    mcp_span = spans["mcp.get_credit_report"]
    db_span = spans["db.query:credit_reports"]
    assert mcp_span.parent.span_id == tool_span.context.span_id
    assert db_span.parent.span_id == mcp_span.context.span_id


def test_general_faq_has_no_mcp_or_db_nesting(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "get_tracer", lambda: provider.get_tracer("test"))

    tool_call = _FakeToolCall("call_1", "general_faq", {"topic": "how_it_works"})
    _patch_openai(
        monkeypatch,
        [_FakeMessage(tool_calls=[tool_call]), _FakeMessage(content="Here's how it works.")],
    )

    agent.run_agent("P01", "how does this work?")

    names = {s.name for s in exporter.get_finished_spans()}
    assert "general_faq" in names
    assert not any(n.startswith("mcp.") or n.startswith("db.") for n in names)
