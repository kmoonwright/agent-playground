"""The agent loop, the tool specs, the guardrail, and the keyless rule brain.

The Anthropic loop is tested against a scripted fake client, because the
mechanics that are easy to get wrong -- batching every tool_result into ONE user
message, echoing assistant content verbatim, never sending sampling parameters,
handling stop_reason == "refusal" before reading content -- are exactly the ones
that fail silently against the real API.
"""

from __future__ import annotations

import pytest

import config
import agents.tools as tools


# ---------------------------------------------------------------- fake client


class Block:
    def __init__(self, type_, *, text=None, name=None, input=None, id=None):
        self.type = type_
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class Response:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # Snapshot `messages`: the loop mutates one list in place, so storing the
        # reference would make every recorded call show the final state.
        recorded = dict(kwargs)
        recorded["messages"] = list(kwargs["messages"])
        self.calls.append(recorded)
        if not self.script:
            return Response([Block("text", text="done")])
        return self.script.pop(0)


class FakeAnthropic:
    def __init__(self, messages):
        self.messages = messages


@pytest.fixture
def fake_claude(monkeypatch):
    """Install a scripted Anthropic client; returns a function to load the script."""
    import anthropic

    holder: dict[str, FakeMessages] = {}

    def install(script):
        holder["messages"] = FakeMessages(script)
        monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: FakeAnthropic(holder["messages"]))
        return holder["messages"]

    return install


# ---------------------------------------------------------------- the loop


def test_every_tool_result_goes_back_in_one_user_message(fake_claude, primed_session):
    """Splitting them teaches the model to stop making parallel calls."""
    import agents.llm as llm

    fake = fake_claude(
        [
            Response(
                [
                    Block("text", text="checking both"),
                    Block("tool_use", name="get_now_playing", input={}, id="t1"),
                    Block("tool_use", name="get_now_playing", input={}, id="t2"),
                ],
                stop_reason="tool_use",
            ),
            Response([Block("text", text="queued it")]),
        ]
    )
    dispatch = tools.Dispatch(primed_session, "host")
    result = llm.run_anthropic_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "what's on?"}],
        dispatch=dispatch,
        max_iterations=4,
    )

    second_request = fake.calls[1]["messages"]
    tool_result_messages = [
        m
        for m in second_request
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and all(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_result_messages) == 1, "results must be batched into a single user message"
    assert len(tool_result_messages[0]["content"]) == 2
    assert {b["tool_use_id"] for b in tool_result_messages[0]["content"]} == {"t1", "t2"}
    assert result.text == "checking both\nqueued it"


def test_assistant_content_is_echoed_verbatim(fake_claude, primed_session):
    """Thinking and tool_use blocks must survive the round trip unmodified."""
    import agents.llm as llm

    blocks = [
        Block("thinking", text=""),
        Block("tool_use", name="get_now_playing", input={}, id="t1"),
    ]
    fake = fake_claude(
        [Response(blocks, stop_reason="tool_use"), Response([Block("text", text="ok")])]
    )
    llm.run_anthropic_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        dispatch=tools.Dispatch(primed_session, "host"),
        max_iterations=3,
    )
    echoed = [m for m in fake.calls[1]["messages"] if m["role"] == "assistant"]
    assert echoed[0]["content"] is blocks  # the same object, not a rebuilt copy


def test_sampling_parameters_are_never_sent(fake_claude, primed_session):
    """temperature / top_p / top_k are rejected on claude-opus-5."""
    import agents.llm as llm

    fake = fake_claude([Response([Block("text", text="hi")])])
    llm.run_anthropic_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        dispatch=tools.Dispatch(primed_session, "host"),
        max_iterations=2,
    )
    kwargs = fake.calls[0]
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in kwargs


def test_thinking_is_left_on_and_effort_controls_latency(fake_claude, primed_session):
    """Disabling thinking can make a tool call come back as plain text."""
    import agents.llm as llm

    fake = fake_claude([Response([Block("text", text="hi")])])
    llm.run_anthropic_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        dispatch=tools.Dispatch(primed_session, "host"),
        max_iterations=2,
    )
    kwargs = fake.calls[0]
    assert "thinking" not in kwargs, "omitting `thinking` keeps adaptive thinking on"
    assert kwargs["output_config"] == {"effort": llm.EFFORT}


def test_refusal_is_handled_before_content_is_read(fake_claude, primed_session):
    """stop_reason == 'refusal' arrives as HTTP 200 with empty or partial content."""
    import agents.llm as llm

    fake_claude([Response([], stop_reason="refusal")])
    result = llm.run_anthropic_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        dispatch=tools.Dispatch(primed_session, "host"),
        max_iterations=3,
    )
    assert result.refused is True
    assert "declined" in result.text


def test_subagent_gets_one_nudge_then_gives_up(fake_claude, primed_session):
    import agents.llm as llm
    import agents.prompt as prompts

    fake = fake_claude(
        [
            Response([Block("text", text="thinking out loud")]),
            Response([Block("text", text="still not calling report")]),
        ]
    )
    dispatch = tools.Dispatch(primed_session, "crate_digger")
    result = llm.run_anthropic_agent(
        role="crate_digger",
        system="s",
        messages=[{"role": "user", "content": "find something"}],
        dispatch=dispatch,
        max_iterations=5,
    )
    final_messages = fake.calls[-1]["messages"]
    nudges = [m for m in final_messages if m.get("content") == prompts.NUDGE]
    assert len(nudges) == 1, "exactly one nudge, then stop"
    assert result.terminal is None
    assert len(fake.calls) == 2


def test_terminal_tool_ends_the_loop_immediately(fake_claude, primed_session, local_source):
    import agents.llm as llm
    from schema import Candidates

    track_id = local_source.scan()[0].track_id
    fake = fake_claude(
        [
            Response(
                [
                    Block(
                        "tool_use",
                        name="report",
                        input={"candidates": [{"track_id": track_id, "reason": "122 BPM, cached"}]},
                        id="t1",
                    )
                ],
                stop_reason="tool_use",
            ),
            Response([Block("text", text="should never be reached")]),
        ]
    )
    dispatch = tools.Dispatch(primed_session, "crate_digger")
    result = llm.run_anthropic_agent(
        role="crate_digger",
        system="s",
        messages=[{"role": "user", "content": "find something"}],
        dispatch=dispatch,
        max_iterations=5,
    )
    assert isinstance(result.terminal, Candidates)
    assert len(fake.calls) == 1, "the loop must stop as soon as the terminal tool fires"


def test_iteration_budget_is_respected(fake_claude, primed_session):
    import agents.llm as llm

    fake = fake_claude(
        [
            Response(
                [Block("tool_use", name="get_now_playing", input={}, id=f"t{i}")],
                stop_reason="tool_use",
            )
            for i in range(10)
        ]
    )
    result = llm.run_anthropic_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "loop forever"}],
        dispatch=tools.Dispatch(primed_session, "host"),
        max_iterations=3,
    )
    assert len(fake.calls) == 3
    assert result.iterations == 3


def test_unknown_tool_becomes_an_observation_not_an_exception(primed_session):
    dispatch = tools.Dispatch(primed_session, "host")
    observation = dispatch.run("launch_missiles", {})
    assert "Unknown tool" in observation
    assert "get_now_playing" in observation


def test_bad_arguments_become_a_readable_observation(primed_session):
    dispatch = tools.Dispatch(primed_session, "host")
    observation = dispatch.run("set_target_bpm", {"bpm": 9999})
    assert "Invalid arguments" in observation
    assert "bpm" in observation


# ---------------------------------------------------------------- specs


def test_one_spec_adapts_to_both_providers():
    for role, specs in tools.TOOLS_BY_ROLE.items():
        anthro = tools.to_anthropic(specs)
        openai = tools.to_openai(specs)
        assert [t["name"] for t in anthro] == [t["function"]["name"] for t in openai]
        for a, o in zip(anthro, openai):
            assert set(a) == {"name", "description", "input_schema"}
            assert o["type"] == "function"
            assert set(o["function"]) == {"name", "description", "parameters"}
            assert a["input_schema"] is o["function"]["parameters"]
            schema = a["input_schema"]
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            # Descriptions say *when* to call, which is what drives tool choice.
            assert len(a["description"]) > 80, f"{role}/{a['name']} description is too thin"


def test_every_role_has_a_dispatch_path(primed_session):
    for role, specs in tools.TOOLS_BY_ROLE.items():
        dispatch = tools.Dispatch(primed_session, role, delegate=lambda n, a: "delegated")
        for spec in specs:
            observation = dispatch.run(spec["name"], {})
            assert "not implemented" not in observation, f"{role}/{spec['name']} has no handler"


# ---------------------------------------------------------------- models


def test_model_resolution_precedence(monkeypatch):
    assert config.resolve_model("claude", "host") == "claude-opus-5"
    assert config.resolve_model("claude", "crate_digger") == "claude-haiku-4-5"

    monkeypatch.setenv("DJ_MODEL", "env-all")
    assert config.resolve_model("claude", "host") == "env-all"

    monkeypatch.setenv("DJ_MODEL_HOST", "env-host")
    assert config.resolve_model("claude", "host") == "env-host"
    assert config.resolve_model("claude", "crate_digger") == "env-all"

    config.set_model_overrides({r: None for r in config.ROLES}, "cli-all")
    assert config.resolve_model("claude", "host") == "cli-all"

    config.set_model_overrides({"host": "cli-host"}, "cli-all")
    assert config.resolve_model("claude", "host") == "cli-host"
    assert config.resolve_model("claude", "crate_digger") == "cli-all"


def test_each_role_can_be_pointed_at_a_different_model():
    """The point of the per-role table: the sub-agents need not share the host's model."""
    config.set_model_overrides({"crate_digger": "cheap-model"}, None)
    assert config.resolve_model("claude", "crate_digger") == "cheap-model"
    assert config.resolve_model("claude", "host") == config.DEFAULT_MODELS[("claude", "host")]


def test_history_never_accumulates_stale_state_blocks(fake_claude, primed_session):
    """The state block is regenerated each turn; carrying old ones forward would
    feed the model several contradictory deck snapshots at once."""
    import agents.llm as llm

    brain = llm.LlmBrain("claude", primed_session, lambda _: None)

    for turn in range(3):
        fake_claude([Response([Block("text", text=f"turn {turn}")])])
        brain.handle(f"utterance {turn}")

    def text_of(message: dict) -> str:
        content = message.get("content")
        return content if isinstance(content, str) else ""

    assert not [m for m in brain.history if text_of(m).startswith("Current state")], (
        "no state block should survive into history"
    )
    # The utterances themselves must survive, or history is not doing its job.
    assert sum(1 for m in brain.history if text_of(m).startswith("utterance")) == 3


def test_a_hallucinated_tool_name_is_not_recorded_as_a_call(primed_session):
    dispatch = tools.Dispatch(primed_session, "host")
    dispatch.run("get_now_playing", {})
    dispatch.run("launch_missiles", {})
    assert dispatch.calls == ["get_now_playing"]


# ---------------------------------------------------------------- OpenAI runner
# The two runners are deliberately separate implementations (see llm.py's module
# docstring), which means the OpenAI copy needs its own coverage or the
# duplication is untested duplication.


class OaiFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class OaiToolCall:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.type = "function"
        self.function = OaiFunction(name, arguments)


class OaiMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class OaiResponse:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        recorded = dict(kwargs)
        recorded["messages"] = list(kwargs["messages"])
        self.calls.append(recorded)
        if not self.script:
            return OaiResponse(OaiMessage(content="done"))
        return self.script.pop(0)


@pytest.fixture
def fake_openai(monkeypatch):
    import openai

    holder: dict[str, FakeCompletions] = {}

    def install(script):
        holder["c"] = FakeCompletions(script)
        chat = type("Chat", (), {"completions": holder["c"]})()
        monkeypatch.setattr(
            openai, "OpenAI", lambda *a, **k: type("Client", (), {"chat": chat})()
        )
        return holder["c"]

    return install


def test_openai_runner_puts_one_tool_message_per_call(fake_openai, primed_session):
    """OpenAI's shape is one `role: tool` message each -- not Anthropic's batch."""
    import agents.llm as llm

    fake = fake_openai(
        [
            OaiResponse(
                OaiMessage(
                    content="checking",
                    tool_calls=[
                        OaiToolCall("c1", "get_now_playing", "{}"),
                        OaiToolCall("c2", "get_now_playing", "{}"),
                    ],
                )
            ),
            OaiResponse(OaiMessage(content="queued it")),
        ]
    )
    result = llm.run_openai_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "what's on?"}],
        dispatch=tools.Dispatch(primed_session, "host"),
        max_iterations=4,
    )
    final = fake.calls[-1]["messages"]
    tool_messages = [m for m in final if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    assert final[0]["role"] == "system", "the system prompt is prepended for OpenAI"
    assert result.text == "checking\nqueued it"


def test_openai_runner_reports_unparseable_arguments_as_an_observation(
    fake_openai, primed_session
):
    import agents.llm as llm

    fake = fake_openai(
        [
            OaiResponse(
                OaiMessage(tool_calls=[OaiToolCall("c1", "get_now_playing", "{not json")])
            ),
            OaiResponse(OaiMessage(content="ok")),
        ]
    )
    llm.run_openai_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        dispatch=tools.Dispatch(primed_session, "host"),
        max_iterations=3,
    )
    observation = [m for m in fake.calls[-1]["messages"] if m.get("role") == "tool"][0]
    assert "Could not parse your arguments as JSON" in observation["content"]


def test_openai_runner_respects_the_iteration_budget(fake_openai, primed_session):
    import agents.llm as llm

    fake = fake_openai(
        [
            OaiResponse(
                OaiMessage(tool_calls=[OaiToolCall(f"c{i}", "get_now_playing", "{}")])
            )
            for i in range(10)
        ]
    )
    result = llm.run_openai_agent(
        role="host",
        system="s",
        messages=[{"role": "user", "content": "loop"}],
        dispatch=tools.Dispatch(primed_session, "host"),
        max_iterations=3,
    )
    assert len(fake.calls) == 3
    assert result.iterations == 3


def test_both_runners_reach_the_same_terminal_tool(fake_claude, fake_openai, primed_session, local_source):
    """One spec, two providers, same submit-and-stop contract."""
    import agents.llm as llm
    from schema import Candidates

    track_id = local_source.scan()[0].track_id
    args = {"candidates": [{"track_id": track_id, "reason": "cached, on target"}]}

    fake_claude([Response([Block("tool_use", name="report", input=args, id="t1")],
                          stop_reason="tool_use")])
    claude_result = llm.run_anthropic_agent(
        role="crate_digger",
        system="s",
        messages=[{"role": "user", "content": "find"}],
        dispatch=tools.Dispatch(primed_session, "crate_digger"),
        max_iterations=5,
    )

    import json as _json

    fake_openai([OaiResponse(OaiMessage(
        tool_calls=[OaiToolCall("c1", "report", _json.dumps(args))]
    ))])
    openai_result = llm.run_openai_agent(
        role="crate_digger",
        system="s",
        messages=[{"role": "user", "content": "find"}],
        dispatch=tools.Dispatch(primed_session, "crate_digger"),
        max_iterations=5,
    )

    assert isinstance(claude_result.terminal, Candidates)
    assert isinstance(openai_result.terminal, Candidates)
    assert claude_result.terminal == openai_result.terminal
