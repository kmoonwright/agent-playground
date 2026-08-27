"""The scheduling state machine: prefetch, stage, arm, flip, and its failure paths.

Everything here drives `session.tick()` by hand and pumps the mixer with
`process()`, exactly as the offline renderer does, so there is no audio device,
no thread, and no wall-clock waiting.
"""

from __future__ import annotations

import time

import config
import music.library as library
from session import MAX_STAGE_ATTEMPTS, STAGE_DEADLINE_S, QueuedTrack


def _pump(session, seconds: float) -> None:
    """Advance the audio clock, ticking the conductor between blocks."""
    frames = int(seconds * config.SR)
    done = 0
    while done < frames:
        n = min(config.BLOCKSIZE, frames - done)
        session.mixer.process(n)
        done += n
        session.tick()


def _load_synchronously(session) -> None:
    """Run whatever the conductor asked the loader for, on this thread."""
    from queue import Empty

    while True:
        try:
            job = session._q_load.get_nowait()
        except Empty:
            return
        session._load_one(job)


def test_a_full_transition_stages_arms_flips_and_credits(primed_session, analysed_track, monkeypatch):
    session = primed_session
    incoming, record = analysed_track

    # Deck A is a 12s fixture track, so shrink the leads to fit inside it.
    monkeypatch.setattr(config, "PREFETCH_LEAD_S", 12.5)
    monkeypatch.setattr(config, "LOAD_DEADLINE_SLACK_S", 1.0)

    session.tool_set_target_bpm({"bpm": record.bpm_used, "ramp_bars": 1})
    assert "Queued" in session.tool_queue_track(
        {"track_id": incoming.track_id, "crossfade_bars": 1, "cue_seconds": 0.0}
    )

    # 1. Prefetch: the conductor asks for the load, and staging follows it.
    session.tick()
    _load_synchronously(session)
    session.tick()
    assert session._state.staged_track_id == incoming.track_id

    # 2. The stage only reaches deck B when the callback drains the command, and
    # only becomes visible on the next snapshot publish.
    from audio.mixer import SNAPSHOT_EVERY_BLOCKS

    for _ in range(SNAPSHOT_EVERY_BLOCKS):
        session.mixer.process(config.BLOCKSIZE)
    assert session.snapshot().b_staged

    # 3. Arm, then run the crossfade out to the flip.
    _pump(session, 13.0)
    snap = session.snapshot()
    assert snap.flips == 1, "the crossfade should have completed"
    assert snap.a_track_id == incoming.track_id

    # 4. Both tracks are credited, in the order they sounded.
    session.tick()
    entries = library.read_credits()
    assert [e.track_id for e in entries][-1] == incoming.track_id
    assert len(entries) == 2
    assert session._state.now_playing is not None
    assert session._state.now_playing.track_id == incoming.track_id
    assert session._state.queue == []
    assert session.snapshot().callback_errors == 0


def test_a_stage_that_never_reaches_deck_b_is_retried(primed_session, analysed_track):
    """`staged_track_id` is set on posting, so a lost command used to wedge the set."""
    session = primed_session
    incoming, _ = analysed_track
    said: list[str] = []
    session.say = said.append

    with session._lock:
        session._state.staged_track_id = incoming.track_id
        session._state.staged_at = time.monotonic() - STAGE_DEADLINE_S - 1.0

    session.tick()
    assert session._state.staged_track_id is None
    assert any("never took" in m for m in said)


def test_an_unloadable_track_is_dropped_after_a_few_attempts(primed_session, analysed_track):
    """Otherwise `_stage` retries at the tick rate for the rest of the run."""
    session = primed_session
    incoming, _ = analysed_track
    said: list[str] = []
    session.say = said.append

    entry = QueuedTrack(track_id=incoming.track_id, crossfade_bars=1, cue_seconds=0.0)
    with session._lock:
        session._state.queue = [entry]

    # The loader is never run, so the track never becomes ready.
    for _ in range(MAX_STAGE_ATTEMPTS + 1):
        session._stage(entry)

    assert session._state.queue == []
    assert any("could not be loaded" in m for m in said)


def test_a_track_that_cannot_be_decoded_is_taken_out_of_play(primed_session, analysed_track):
    """Otherwise the selector picks the same broken file again the moment it is dropped."""
    session = primed_session
    incoming, _ = analysed_track
    session.say = lambda _: None

    entry = QueuedTrack(track_id=incoming.track_id, crossfade_bars=1, cue_seconds=0.0)
    with session._lock:
        session._state.queue = [entry]
    for _ in range(MAX_STAGE_ATTEMPTS + 1):
        session._stage(entry)

    assert "out of play" in session.tool_queue_track({"track_id": incoming.track_id})
    assert incoming.track_id not in {
        t.track_id for t in session.find_candidates(60.0, 200.0, limit=5)
    }


def test_a_track_whose_analysed_tempo_is_unreachable_is_refused_at_stage_time(
    primed_session, analysed_track
):
    """The queue-time gate runs on metadata BPM, which is often absent."""
    session = primed_session
    incoming, record = analysed_track
    said: list[str] = []
    session.say = said.append

    with session._lock:
        session._state.target_bpm = record.bpm_used * 1.5  # far outside +/-8%
    session._ready_put(incoming.track_id, record, session.mixer.a.buf)

    entry = QueuedTrack(track_id=incoming.track_id, crossfade_bars=1, cue_seconds=0.0)
    with session._lock:
        session._state.queue = [entry]
    session._stage(entry)

    assert session._state.staged_track_id is None
    assert session._state.queue == []
    assert any("outside the +/-8% window" in m for m in said)


def test_a_staged_track_is_not_dropped_by_a_tempo_change(primed_session, analysed_track):
    """Dropping it would not stop it sounding, only desync the session from the mixer."""
    session = primed_session
    incoming, record = analysed_track
    said: list[str] = []
    session.say = said.append

    with session._lock:
        session._state.queue = [
            QueuedTrack(track_id=incoming.track_id, crossfade_bars=1, cue_seconds=0.0)
        ]
        session._state.staged_track_id = incoming.track_id

    dropped = session._drop_unplayable(record.bpm_used * 1.5)
    assert dropped == []
    assert [q.track_id for q in session._state.queue] == [incoming.track_id]
    assert any("already on deck B" in m for m in said)


def test_the_end_of_the_set_is_announced_and_queueing_is_refused(primed_session, analysed_track):
    """`_tick` used to return silently forever while `queue_track` promised 0s."""
    session = primed_session
    incoming, _ = analysed_track
    said: list[str] = []
    session.say = said.append

    session._underrun(0.0)  # what the conductor does when nothing is staged
    session.mixer.process(config.BLOCKSIZE)  # the callback picks up the FadeOut
    assert session.mixer._fade_gain is not None
    _pump(session, 9.0)
    assert not session.snapshot().a_active

    session.tick()
    assert any("the set has ended" in m for m in said)
    observation = session.tool_queue_track({"track_id": incoming.track_id})
    assert "already ended" in observation


def test_a_flip_with_unreadable_metadata_still_moves_now_playing(primed_session, analysed_track):
    """A stale `now_playing` would credit the next flip to the wrong track."""
    session = primed_session
    incoming, _ = analysed_track
    said: list[str] = []
    session.say = said.append

    library.meta_path(incoming.track_id).unlink()
    snap = session.snapshot()
    session._last_flips = snap.flips - 1
    object.__setattr__(snap, "a_track_id", incoming.track_id)
    session._detect_flip(snap)

    assert session._state.now_playing is not None
    assert session._state.now_playing.track_id == incoming.track_id
    assert any("no credit line was written" in m for m in said)
