"""Plan-building for generate_sessions.py — no network, no MCP except the one mock-path turn."""

import config
import generate_sessions


def test_build_session_plans_cycles_journeys_and_names_session_ids():
    plans = generate_sessions.build_session_plans(len(generate_sessions.JOURNEYS) + 2)
    names = [p["name"] for p in plans]
    assert names[: len(generate_sessions.JOURNEYS)] == [j["name"] for j in generate_sessions.JOURNEYS]
    assert names[-2:] == [generate_sessions.JOURNEYS[0]["name"], generate_sessions.JOURNEYS[1]["name"]]
    assert [p["session_id"] for p in plans] == [
        f"session-{p['name']}-{idx:02d}" for idx, p in enumerate(plans)
    ]
    assert len({p["session_id"] for p in plans}) == len(plans)


def test_each_journey_is_a_four_turn_ax_workflow():
    names = [j["name"] for j in generate_sessions.JOURNEYS]
    assert names == [
        "span-onboarding",
        "eval-lifecycle",
        "reliability-loop",
        "mcp-tracing",
        "compound-deep",
        "coverage-boundary",
    ]
    for journey in generate_sessions.JOURNEYS:
        assert len(journey["messages"]) == 4
        assert all(message.strip() for message in journey["messages"])


async def test_run_session_completes_one_turn_on_the_mock_path(monkeypatch):
    monkeypatch.setattr(config, "openai_api_key", lambda: "")
    plan = {
        "session_id": "session-span-onboarding-test",
        "name": "span-onboarding",
        "messages": ["What are the eleven OpenInference span kinds?"],
    }
    turns = await generate_sessions.run_session(plan, None)
    assert len(turns) == 1
    assert turns[0]["error"] is None
    assert turns[0]["answer"]


def test_journeys_cover_the_shapes_a_demo_needs_to_show():
    all_messages = " ".join(m for j in generate_sessions.JOURNEYS for m in j["messages"])
    assert "eleven OpenInference span kinds" in all_messages
    assert "EMBEDDING and RETRIEVER" in all_messages
    assert "online vs offline" in all_messages
    assert "Enterprise pricing" in all_messages
    assert "MCPInstrumentor" in all_messages
    assert "eight-step improvement loop" in all_messages
