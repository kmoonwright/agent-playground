"""The keyless selector: the routing table, and that it uses the same tools."""

from __future__ import annotations

import pytest

import agents.brain as brain


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("warm dubby house with vocals", ["vocals", "dub", "house"]),
        ("some hip hop please", ["hip_hop"]),
        ("anything ambient", ["ambient"]),
        ("just play music", []),
    ],
)
def test_tag_extraction(text, expected):
    assert brain.extract_tags(text) == expected


def test_bpm_request_routes_to_set_target_bpm(primed_session):
    the_brain = brain.RuleBrain(primed_session, lambda _: None)
    reply = the_brain._route("drop to 112 bpm")
    assert "112" in reply or "Ramping" in reply
    assert primed_session.target_bpm() == pytest.approx(112.0, abs=0.6)


def test_skip_request_routes_to_skip(primed_session):
    the_brain = brain.RuleBrain(primed_session, lambda _: None)
    assert "Nothing is queued" in the_brain._route("skip this one")
    assert the_brain.dispatch.calls == ["skip"]


def test_status_request_routes_to_now_playing(primed_session):
    the_brain = brain.RuleBrain(primed_session, lambda _: None)
    assert "Now playing:" in the_brain._route("what's playing?")


def test_faster_nudges_the_target_up(primed_session):
    the_brain = brain.RuleBrain(primed_session, lambda _: None)
    before = primed_session.target_bpm()
    the_brain._route("more upbeat")
    assert primed_session.target_bpm() > before


def test_a_plain_request_queues_something(primed_session):
    the_brain = brain.RuleBrain(primed_session, lambda _: None)
    reply = the_brain._route("play something")
    assert reply.startswith("Queued") or "already" in reply
    if reply.startswith("Queued"):
        with primed_session._lock:
            assert len(primed_session._state.queue) == 1


def test_autopilot_messages_are_handled_without_a_user(primed_session):
    the_brain = brain.RuleBrain(primed_session, lambda _: None)
    reply = the_brain._route("[autopilot] the queue is empty, 40s left")
    assert reply.startswith("Queued") or "already" in reply or "could not find" in reply


def test_rule_brain_uses_the_same_dispatch_as_the_llm_brains(primed_session):
    """One implementation of every tool, so the keyless path is not a stub."""
    the_brain = brain.RuleBrain(primed_session, lambda _: None)
    assert the_brain.dispatch.session is primed_session
    assert the_brain.dispatch.role == "host"
    the_brain._route("what's playing?")
    assert the_brain.dispatch.calls == ["get_now_playing"]
