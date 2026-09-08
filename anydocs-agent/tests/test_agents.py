"""Host/librarian tool-calling loop against a scripted fake OpenAI client —
same pattern as dj-agent's fake_claude fixture: fake the SDK client class
itself, script a sequence of responses, assert on what gets sent and
returned. Exercises the "real" (has-a-key) code path deterministically,
without a network call.
"""

import json

import pytest

import config
import mcp_client
from agents import host


class FakeFunctionCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments):
        self.id = "call_1"
        self.function = FakeFunctionCall(name, json.dumps(arguments))


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = self.script.pop(0)
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


class FakeOpenAI:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


@pytest.fixture
def fake_openai(monkeypatch):
    import openai

    def install(messages: list[FakeMessage]) -> FakeCompletions:
        completions = FakeCompletions(messages)
        monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: FakeOpenAI(completions))
        monkeypatch.setattr(config, "openai_api_key", lambda: "fake-key-for-tests")
        return completions

    return install


async def test_guardrail_strips_a_fabricated_citation(fake_openai):
    fake_openai(
        [
            FakeMessage(tool_calls=[FakeToolCall("ask_librarian", {"query": "span kinds"})]),
            FakeMessage(tool_calls=[FakeToolCall("search_docs", {"query": "span kinds"})]),
            FakeMessage(content="(no second hop needed)"),  # host decides to stop hopping
            FakeMessage(content="Spans have kinds. [totally-made-up-doc]"),
        ]
    )
    async with mcp_client.open_session() as session:
        answer = await host.run(session, "test-session", "what are span kinds?")
    assert "totally-made-up-doc" not in answer
    assert "[totally-made-up-doc]" not in answer


async def test_host_skips_search_when_the_model_declines(fake_openai):
    # First call: host decides whether to delegate (declines, no tool_calls).
    # Second call: host drafts the final answer directly.
    fake_openai([FakeMessage(content="(not delegating)"), FakeMessage(content="Hey there!")])
    async with mcp_client.open_session() as session:
        answer = await host.run(session, "test-session", "hello")
    assert answer == "Hey there!"


async def test_host_can_delegate_twice_for_a_compound_question(fake_openai):
    completions = fake_openai(
        [
            FakeMessage(tool_calls=[FakeToolCall("ask_librarian", {"query": "span kinds"})]),
            FakeMessage(tool_calls=[FakeToolCall("search_docs", {"query": "span kinds"})]),
            FakeMessage(
                tool_calls=[
                    FakeToolCall("ask_librarian", {"query": "online vs offline evaluators"})
                ]
            ),
            FakeMessage(tool_calls=[FakeToolCall("search_docs", {"query": "online vs offline evaluators"})]),
            FakeMessage(content="(no third hop needed)"),
            FakeMessage(content="Combined answer. [span-kinds] [evaluators-online-vs-offline]"),
        ]
    )
    async with mcp_client.open_session() as session:
        answer = await host.run(session, "test-session", "a compound question")
    assert "[span-kinds]" in answer
    assert "[evaluators-online-vs-offline]" in answer
    assert len(completions.calls) == 6


async def test_host_stops_hopping_at_max_hops(fake_openai):
    # A fake client scripted to always want another hop. If host.run's loop
    # bound were missing or off by one, this would try to pop a script
    # entry that doesn't exist and raise, instead of completing cleanly.
    script = []
    for i in range(host.MAX_HOPS):
        script.append(FakeMessage(tool_calls=[FakeToolCall("ask_librarian", {"query": f"topic {i}"})]))
        script.append(FakeMessage(tool_calls=[FakeToolCall("search_docs", {"query": f"topic {i}"})]))
    script.append(FakeMessage(content="final answer, no citations"))
    completions = fake_openai(script)

    async with mcp_client.open_session() as session:
        answer = await host.run(session, "test-session", "keeps wanting to search forever")

    assert answer == "final answer, no citations"
    assert len(completions.calls) == host.MAX_HOPS * 2 + 1
