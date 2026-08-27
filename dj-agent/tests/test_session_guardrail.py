"""Session tools, the varispeed gate, and the transition guardrail."""

from __future__ import annotations

import pytest

import config
import music.library as library
from schema import clamp_rate


def test_clamp_rate_bounds_the_varispeed_window():
    assert clamp_rate(1.0) == (1.0, None)
    low, note = clamp_rate(0.5)
    assert low == config.RATE_MIN and "floor" in note
    high, note = clamp_rate(2.0)
    assert high == config.RATE_MAX and "ceiling" in note


def test_queue_rejects_a_track_too_far_off_tempo(primed_session, local_source):
    """The +/-8% window is the hard constraint; taste does not override it."""
    session = primed_session
    with session._lock:
        session._state.target_bpm = 160.0

    track = next(t for t in local_source.scan() if t.bpm == 120.0)
    library.save_track(track)
    observation = session.tool_queue_track({"track_id": track.track_id})
    assert "outside the +/-8% varispeed window" in observation
    with session._lock:
        assert session._state.queue == []


def test_queue_accepts_a_track_inside_the_window(primed_session, analysed_track):
    session = primed_session
    incoming, record = analysed_track
    with session._lock:
        session._state.target_bpm = record.bpm_used

    observation = session.tool_queue_track({"track_id": incoming.track_id})
    assert observation.startswith("Queued")
    with session._lock:
        assert [q.track_id for q in session._state.queue] == [incoming.track_id]

    # Queueing it twice is a no-op with an explanation, not a duplicate.
    again = session.tool_queue_track({"track_id": incoming.track_id})
    assert "already in the queue" in again


def test_queue_reports_an_unknown_id_usefully(primed_session):
    observation = primed_session.tool_queue_track({"track_id": "0" * 12})
    assert "No track with id" in observation
    assert "Search first" in observation


def test_set_target_bpm_clamps_and_says_so(primed_session):
    observation = primed_session.tool_set_target_bpm({"bpm": 179})
    assert "clamp" in observation.lower()
    # The effective tempo must land inside the window, not at the request.
    assert primed_session.target_bpm() <= 120.0 * config.RATE_MAX + 0.01


def test_set_target_bpm_defers_during_a_crossfade(primed_session, monkeypatch):
    """Both decks are already locked to one tempo mid-blend; ramping would break it."""
    session = primed_session
    snap = session.snapshot()
    monkeypatch.setattr(
        session, "snapshot", lambda: snap.__class__(**{**snap.__dict__, "xfade_len": 1000})
    )
    observation = session.tool_set_target_bpm({"bpm": 118})
    assert "Deferred" in observation
    with session._lock:
        assert session._state.pending_target_bpm == 118.0


def test_skip_with_nothing_queued_says_what_to_do(primed_session):
    observation = primed_session.tool_skip({})
    assert "Nothing is queued" in observation


def test_guardrail_clamps_an_impossible_plan(primed_session, analysed_track):
    """The planner is advisory; the validator owns what the mixer may do."""
    session = primed_session
    incoming, record = analysed_track

    plan = session.validate_plan(
        {
            "crossfade_bars": 32,
            "cue_seconds": 9_999.0,  # way past the end of the track
            "target_bpm": 175.0,  # unreachable for either deck
            "note": "big long blend",
        },
        incoming.track_id,
    )

    assert plan.was_corrected
    assert config.RATE_MIN <= plan.rate_a <= config.RATE_MAX
    assert config.RATE_MIN <= plan.rate_b <= config.RATE_MAX
    assert plan.cue_seconds < record.duration_s
    assert any("varispeed" in c for c in plan.corrections)
    assert plan.note == "big long blend"  # the model's reasoning survives
    # Both rates have to describe the *same* tempo, which is what an unrolled
    # clamp pass got wrong: it reported success with the decks 19% apart.
    assert plan.rate_a * 120.0 == pytest.approx(plan.rate_b * record.bpm_used, rel=1e-3)
    assert plan.target_bpm == pytest.approx(plan.rate_b * record.bpm_used, rel=1e-3)


def test_guardrail_leaves_a_reasonable_plan_alone(primed_session, analysed_track):
    session = primed_session
    incoming, record = analysed_track

    plan = session.validate_plan(
        {
            "crossfade_bars": 8,
            "cue_seconds": 0.0,
            "target_bpm": round((record.bpm_used + 120.0) / 2, 1),
            "note": "split the difference",
        },
        incoming.track_id,
    )
    assert plan.corrections == []
    assert plan.crossfade_bars == 8


def test_guardrail_rejects_an_unanalysed_track(primed_session):
    from session import ToolError

    with pytest.raises(ToolError, match="not been analysed"):
        primed_session.validate_plan(
            {"crossfade_bars": 8, "cue_seconds": 0.0, "target_bpm": 120.0}, "unknown"
        )


def test_now_playing_reports_real_state(primed_session):
    observation = primed_session.tool_get_now_playing()
    assert "Now playing:" in observation
    assert "Target BPM:" in observation
    assert "Queue is empty." in observation
    assert "Audio dropouts: 0" in observation


def test_a_tempo_change_drops_queued_tracks_that_no_longer_fit(primed_session, analysed_track):
    """And bumps the generation, so an in-flight load for one is discarded."""
    session = primed_session
    incoming, record = analysed_track

    with session._lock:
        session._state.target_bpm = record.bpm_used
    assert session.tool_queue_track({"track_id": incoming.track_id}).startswith("Queued")
    generation_before = session.target_generation()

    # Move the tempo far enough that the queued track cannot be beatmatched.
    observation = session.tool_set_target_bpm({"bpm": record.bpm_used * 0.75})
    assert "no longer fit" in observation
    assert session.queue_depth() == 0
    # Which is what discards a load already in flight for the dropped track.
    assert session.target_generation() > generation_before


def test_a_small_tempo_change_keeps_the_queue(primed_session, analysed_track):
    session = primed_session
    incoming, record = analysed_track

    with session._lock:
        session._state.target_bpm = record.bpm_used
    session.tool_queue_track({"track_id": incoming.track_id})
    generation_before = session.target_generation()

    session.tool_set_target_bpm({"bpm": record.bpm_used * 1.01})
    assert session.queue_depth() == 1
    assert session.target_generation() == generation_before


def test_guardrail_refuses_to_pretend_two_far_apart_tempos_match():
    """100 and 140 BPM cannot meet inside +/-8%; the plan must say so, not fake it."""
    from schema import resolve_target_bpm

    target, rate_a, rate_b, corrections = resolve_target_bpm(120.0, 100.0, 140.0)
    assert config.RATE_MIN <= rate_b <= config.RATE_MAX  # deck B stays playable
    assert rate_a == pytest.approx(config.RATE_MAX)  # deck A goes as far as it can
    assert 100.0 * rate_a != pytest.approx(target, rel=1e-3)  # and still falls short
    assert any("will not beat-match" in c for c in corrections)


def test_guardrail_converges_when_one_clamp_moves_the_other_deck():
    """A single clamp pass leaves the decks at different tempos; this must not."""
    from schema import resolve_target_bpm

    for bpm_a, bpm_b, requested in [(120.0, 128.0, 180.0), (128.0, 120.0, 60.0), (120.0, 120.0, 95.0)]:
        target, rate_a, rate_b, _ = resolve_target_bpm(requested, bpm_a, bpm_b)
        assert config.RATE_MIN <= rate_a <= config.RATE_MAX
        assert config.RATE_MIN <= rate_b <= config.RATE_MAX
        assert bpm_a * rate_a == pytest.approx(target)
        assert bpm_b * rate_b == pytest.approx(target)


def test_queueing_an_untagged_track_does_not_crash_on_its_missing_bpm(primed_session, crate):
    """The common local-crate case: no BPM tag, no analysis yet, so bpm is None."""
    import music.library as library
    from music.local import LocalSource

    track = [t for t in LocalSource(crate).scan() if t.bpm is None][0]
    library.save_track(track)
    observation = primed_session.tool_queue_track({"track_id": track.track_id})
    assert observation.startswith("Queued")
    assert "tempo not known yet" in observation
