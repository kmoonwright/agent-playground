"""DJSession: the shared state, the loader thread, the conductor, and every tool.

Thread map (see mixer.py for the callback rule):

  T1  REPL            -- dj.py
  T2  brain           -- brain.py, plus a small sub-agent pool
  T3  loader          -- _loader_loop below: network + librosa
  T4  audio callback  -- mixer.py. Never touched from here.
  T5  conductor       -- _conductor_loop below: scheduling, credits, spans

All mutable session state lives in `self._state` under one RLock. No I/O and no
span export ever happens while that lock is held: the loader copies out what it
needs, works, then re-acquires to publish.

Deck feeding is deterministic and lives on the conductor. The LLM is never in
the deadline path: if the queue is empty the conductor *asks* the brain for a
track 45 seconds early and re-asks every 20 seconds, and if no answer ever
arrives it fades deck A out over 4 bars rather than stalling the callback.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any, Callable, Sequence

import numpy as np

import audio.analysis as analysis_mod
import config
import instrumentation as ins
import music.library as library
from audio.mixer import (
    FadeOut,
    LoadDeckB,
    Mixer,
    Snapshot,
    SetRateRamp,
    StartXfade,
    bars_to_samples,
    equal_power_curves,
    linear_fade_out,
    next_bar_after,
    rate_ramp,
)
from schema import (
    Analysis,
    GetAnalysisArgs,
    InspectTrackArgs,
    QueueTrackArgs,
    SearchSourceArgs,
    SetTargetBpmArgs,
    SkipArgs,
    ValidatedPlan,
    bpm_within_varispeed,
    clamp_rate,
    playable_window,
    validate_plan,
)
from music.source import SearchQuery, Source, SourceError, Track

READY_CACHE_MAX = 3
CONDUCTOR_TICK_S = 0.25
AUTOPILOT_COOLDOWN_S = 20.0
# How long deck B may take to appear in the snapshot after LoadDeckB is posted
# before the conductor gives up and re-stages. A stage is one deque append plus
# an already-decoded buffer, so anything near this is a real failure.
STAGE_DEADLINE_S = 5.0
# Load requests per queued track before it is dropped. Without a bound an
# unloadable track retries at the tick rate for the rest of the set.
MAX_STAGE_ATTEMPTS = 3


@dataclass
class QueuedTrack:
    track_id: str
    crossfade_bars: int
    cue_seconds: float
    plan_note: str = ""


@dataclass
class NowPlaying:
    track_id: str
    title: str
    artist: str
    bpm_used: float
    started_at: float


@dataclass
class SessionState:
    target_bpm: float = config.DEFAULT_TARGET_BPM
    queue: list[QueuedTrack] = field(default_factory=list)
    now_playing: NowPlaying | None = None
    staged_track_id: str | None = None
    staged_at: float = 0.0
    armed_track_id: str | None = None
    pending_target_bpm: float | None = None
    generation: int = 0
    underruns: int = 0
    stage_attempts: dict[str, int] = field(default_factory=dict)
    ended: bool = False


@dataclass
class LoadJob:
    track: Track
    generation: int
    ctx: Any
    purpose: str = "queue"


class ToolError(RuntimeError):
    """A tool failure that should reach the agent as text, never as a traceback."""


class DJSession:
    def __init__(
        self,
        *,
        mixer: Mixer,
        sources: dict[str, Source],
        say: Callable[[str], None],
        set_id: str,
    ) -> None:
        self.mixer = mixer
        self.sources = sources
        self.say = say
        self.set_id = set_id

        self._state = SessionState()
        self._lock = threading.RLock()

        self.q_user: Queue[str] = Queue(maxsize=8)
        self._q_load: Queue[LoadJob] = Queue(maxsize=4)

        self._ready: dict[str, tuple[Analysis, np.ndarray]] = {}
        self._ready_order: list[str] = []
        self._ready_lock = threading.Lock()

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_flips = 0
        self._last_dropped_xfades = 0
        self._last_autopilot_at = 0.0
        self._faded_out = False
        self._transition_span = None
        # Validated plans, keyed by track id, so queue_track can pick up the
        # planner's note and cue point even though the host passes only numbers.
        self._plans: dict[str, ValidatedPlan] = {}
        # Tracks whose audio could not be decoded. Without this the selector
        # picks the same broken file again the moment it is dropped, and the set
        # spends the rest of its life re-downloading and re-failing on it.
        self._unloadable: set[str] = set()

    # ------------------------------------------------------------ lifecycle

    def start(self, *, conductor: bool = True) -> None:
        """Start the background threads.

        `conductor=False` is for the offline renderer, which drives `tick()`
        from its own block loop so the audio clock and the scheduling clock stay
        in step even when rendering faster than realtime.
        """
        targets = [("loader", self._loader_loop)]
        if conductor:
            targets.append(("conductor", self._conductor_loop))
        for name, target in targets:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def tick(self) -> None:
        """One scheduling step. Public so the renderer can drive it."""
        self._tick()

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._state.queue)

    def faded_out(self) -> bool:
        return self._faded_out

    def waiting_for_load(self) -> bool:
        """True when the conductor needs a track that is not analysed yet."""
        with self._lock:
            queue = list(self._state.queue)
        if not queue:
            return False
        return self._ready_get(queue[0].track_id) is None

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._end_transition_span(reason="session stopped")

    # ------------------------------------------------------------ state helpers

    def snapshot(self):
        return self.mixer.snapshot

    def target_bpm(self) -> float:
        with self._lock:
            return self._state.target_bpm

    def _bump_generation(self) -> int:
        with self._lock:
            self._state.generation += 1
            return self._state.generation

    def _ready_put(self, track_id: str, analysis: Analysis, audio: np.ndarray) -> None:
        with self._ready_lock:
            self._ready[track_id] = (analysis, audio)
            if track_id in self._ready_order:
                self._ready_order.remove(track_id)
            self._ready_order.append(track_id)
            while len(self._ready_order) > READY_CACHE_MAX:
                evicted = self._ready_order.pop(0)
                self._ready.pop(evicted, None)

    def _ready_get(self, track_id: str) -> tuple[Analysis, np.ndarray] | None:
        with self._ready_lock:
            return self._ready.get(track_id)

    # ------------------------------------------------------------ loader (T3)

    def _loader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q_load.get(timeout=0.25)
            except Empty:
                continue
            try:
                self._load_one(job)
            except Exception as exc:  # never kill the loader
                self.say(f"[loader] {job.track.label}: {exc}")

    def _load_one(self, job: LoadJob) -> None:
        # Re-attach the OTel context captured when the job was queued, or this
        # span becomes a trace root instead of a child of the turn that asked.
        with ins.propagate_to_thread(job.ctx):
            with ins.traced_span(
                "track.load",
                ins.CHAIN,
                input_value=job.track.label,
                attributes={
                    "track_id": job.track.track_id,
                    "source": job.track.source,
                    "purpose": job.purpose,
                },
            ) as span:
                if self._is_stale(job.generation):
                    ins.set_output(span, "discarded: superseded by a newer request")
                    return

                source = self.sources.get(job.track.source)
                if source is None:
                    raise ToolError(f"source {job.track.source!r} is not enabled this run")

                with ins.traced_span(
                    "fetch_and_analyze", ins.CHAIN, attributes={"source": job.track.source}
                ) as an_span:
                    record, audio, cached = library.fetch_and_analyze(job.track, source)
                    ins.set_attributes(
                        an_span,
                        {
                            "cache_hit": cached,
                            "bpm_meta": record.bpm_meta,
                            "bpm_librosa": record.bpm_librosa,
                            "bpm_used": record.bpm_used,
                            "bpm_source": record.bpm_source,
                            "grid_confidence": record.grid_confidence,
                            "grid_source": analysis_mod.grid_source(record),
                            "duration_s": record.duration_s,
                        },
                    )

                if self._is_stale(job.generation):
                    ins.set_output(span, "analysed but discarded: superseded")
                    return

                self._ready_put(job.track.track_id, record, audio)
                ins.set_output(
                    span,
                    {
                        "track": job.track.label,
                        "bpm_used": record.bpm_used,
                        "grid_confidence": record.grid_confidence,
                    },
                )
                self.say(
                    f"[loaded] {job.track.label} — {record.bpm_used:.1f} BPM "
                    f"(grid confidence {record.grid_confidence:.2f})"
                )

    def _is_stale(self, generation: int) -> bool:
        with self._lock:
            return generation < self._state.generation

    def _request_load(self, track: Track, purpose: str) -> bool:
        job = LoadJob(
            track=track,
            generation=self.target_generation(),
            ctx=ins.capture_context(),
            purpose=purpose,
        )
        try:
            self._q_load.put_nowait(job)
            return True
        except Full:
            return False

    def target_generation(self) -> int:
        with self._lock:
            return self._state.generation

    # ------------------------------------------------------------ conductor (T5)

    def _conductor_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # never kill the conductor
                self.say(f"[conductor] {type(exc).__name__}: {exc}")
            self._stop.wait(CONDUCTOR_TICK_S)

    def _tick(self) -> None:
        snap = self.snapshot()
        self.mixer.drain_retired()
        self._detect_flip(snap)

        if not snap.a_active:
            self._note_end_of_set()
            return

        remaining = snap.a_remaining_out_s

        with self._lock:
            queue = list(self._state.queue)
            staged = self._state.staged_track_id
            staged_at = self._state.staged_at
            armed = self._state.armed_track_id

        # 0. Staging deadline: `staged` is set when LoadDeckB is *posted*, and
        # until the callback picks it up it blocks both re-staging and the
        # underrun rescue. If deck B never appeared, forget it and try again.
        if (
            staged is not None
            and not snap.b_staged
            and time.monotonic() - staged_at > STAGE_DEADLINE_S
        ):
            self.say(f"[conductor] deck B never took {staged}; re-staging")
            with self._lock:
                self._state.staged_track_id = None
                self._state.armed_track_id = None
            return

        # 1. Prefetch: get the head of the queue analysed and staged early.
        if staged is None and queue and remaining < config.PREFETCH_LEAD_S:
            self._stage(queue[0])
            return

        # 2. Arm the crossfade once deck B is in the mixer and deck A is running out.
        if staged is not None and armed is None and snap.b_staged and queue:
            entry = queue[0]
            xfade_s = bars_to_samples(entry.crossfade_bars, self.target_bpm()) / config.SR
            if remaining <= xfade_s + config.LOAD_DEADLINE_SLACK_S:
                self._arm(entry, snap)
                return

        # 2b. A crossfade the mixer refused (one was already running) would
        # otherwise leave the conductor waiting on a flip that never comes.
        if armed is not None and snap.dropped_xfades > self._last_dropped_xfades:
            self._last_dropped_xfades = snap.dropped_xfades
            self.say(f"[conductor] the mixer refused the crossfade into {armed}; will re-arm")
            with self._lock:
                self._state.armed_track_id = None
            return

        # 3. Autopilot: ask the brain for a track well before we need one.
        if not queue and remaining < config.AUTOPILOT_LEAD_S:
            self._autopilot(remaining)

        # 4. Underrun rescue: fade out rather than stall the callback. Once only
        # per silence episode -- re-posting a FadeOut every tick would both spam
        # the user and keep restarting the fade.
        if (
            not queue
            and staged is None
            and remaining <= 4.0
            and not snap.xfade_len
            and not self._faded_out
        ):
            self._underrun(remaining)

    def _note_end_of_set(self) -> None:
        """Say once that deck A has run out. Nothing restarts a set from here.

        Before this existed `_tick` returned silently forever and `queue_track`
        still answered "it should start in about 0s".
        """
        with self._lock:
            if self._state.ended or not self._faded_out:
                self._state.ended = self._state.ended or self._faded_out
                return
            self._state.ended = True
        self.say("[conductor] the set has ended — deck A is empty. /quit writes the setlist.")

    def _drop_queued(self, track_id: str, why: str, *, unloadable: bool = False) -> None:
        """Take a track out of the queue with one message, never silently."""
        with self._lock:
            if unloadable:
                self._unloadable.add(track_id)
            self._state.queue = [q for q in self._state.queue if q.track_id != track_id]
            self._state.stage_attempts.pop(track_id, None)
            if self._state.staged_track_id == track_id:
                self._state.staged_track_id = None
            if self._state.armed_track_id == track_id:
                self._state.armed_track_id = None
        self.say(f"[conductor] dropped {track_id}: {why}")

    def _detect_flip(self, snap: Snapshot) -> None:
        if snap.flips == self._last_flips:
            return
        self._last_flips = snap.flips

        track_id = snap.a_track_id
        track = library.load_track(track_id) if track_id else None
        record = library.load_analysis(track_id) if track_id else None

        with self._lock:
            previous = self._state.now_playing
            self._state.queue = [q for q in self._state.queue if q.track_id != track_id]
            self._state.staged_track_id = None
            self._state.armed_track_id = None
            self._state.stage_attempts.pop(track_id or "", None)
            pending_bpm = self._state.pending_target_bpm
            self._state.pending_target_bpm = None
            # Deck A really did change, so `now_playing` must change with it even
            # if the sidecars have gone: a stale one would credit the wrong track.
            self._state.now_playing = NowPlaying(
                track_id=track_id or "",
                title=track.title if track else (track_id or "unknown"),
                artist=track.artist if track else "unknown",
                bpm_used=record.bpm_used if record else self._state.target_bpm,
                started_at=time.monotonic(),
            )
        self._faded_out = False

        self._end_transition_span(reason="completed", snap=snap)

        # Credit the track that just took over, at the moment it starts sounding.
        if track is not None and record is not None:
            played = (
                round(time.monotonic() - previous.started_at, 1)
                if previous is not None
                else None
            )
            library.append_credit(library.credit_for(track, record.bpm_used, played))
            self.say(
                f"[now playing] {track.label} — {record.bpm_used:.1f} BPM"
                f"{library.license_flag(track.license_verified)}"
            )
        else:
            self.say(
                f"[now playing] {track_id} — its metadata could not be read, "
                "so no credit line was written for it"
            )

        if pending_bpm is not None:
            self.say(f"[conductor] applying the deferred tempo change to {pending_bpm:.1f} BPM")
            self._apply_target_bpm(pending_bpm, ramp_bars=4)

    def _stage(self, entry: QueuedTrack) -> None:
        """Put an analysed track onto deck B, loading it first if necessary."""
        ready = self._ready_get(entry.track_id)
        if ready is None:
            track = library.load_track(entry.track_id)
            if track is None:
                self._drop_queued(entry.track_id, "is not in the library")
                return
            with self._lock:
                attempts = self._state.stage_attempts.get(entry.track_id, 0) + 1
                self._state.stage_attempts[entry.track_id] = attempts
            if attempts > MAX_STAGE_ATTEMPTS:
                self._drop_queued(
                    entry.track_id,
                    f"could not be loaded after {MAX_STAGE_ATTEMPTS} attempts",
                    unloadable=True,
                )
                return
            self._request_load(track, purpose="stage")
            return

        record, audio = ready
        target = self.target_bpm()
        # The queue-time gate ran against source metadata BPM, which is often
        # absent. This is the first point where the analysed tempo is known, so
        # it is the last chance to refuse a blend that cannot beat-match.
        if not bpm_within_varispeed(record.bpm_used, target):
            self._drop_queued(
                entry.track_id,
                f"analysed at {record.bpm_used:.1f} BPM, outside the +/-8% window "
                f"around the {target:.1f} BPM target",
            )
            return
        rate, _ = clamp_rate(target / record.bpm_used)
        grid = analysis_mod.bar_grid(record)
        cue_samples = entry.cue_seconds * config.SR
        start = next_bar_after(grid, cue_samples)
        if start is None:
            start = int(grid[0]) if grid.size else 0

        self.mixer.post(
            LoadDeckB(
                track_id=entry.track_id,
                buf=audio,
                grid=grid,
                rate=float(rate),
                start_pos=float(start),
            )
        )
        with self._lock:
            self._state.staged_track_id = entry.track_id
            self._state.staged_at = time.monotonic()
        self._faded_out = False

    def _arm(self, entry: QueuedTrack, snap: Snapshot) -> None:
        """Schedule a beat-aligned crossfade on the next bar boundary of deck A."""
        target = self.target_bpm()
        n_xfade = bars_to_samples(entry.crossfade_bars, target)
        gain_a, gain_b = equal_power_curves(n_xfade)
        # The callback finds the bar boundary itself, from deck A's exact position
        # at the moment it applies this. Deck state must not be read from here:
        # the snapshot is up to 8 blocks old, and a delay computed from it lands
        # the crossfade off the beat by however stale it was.
        self.mixer.post(
            StartXfade(delay_frames=0, gain_a=gain_a, gain_b=gain_b, align_to_bar=True)
        )

        # Estimated from the snapshot, only for the message and the span.
        a_bar = next_bar_after(snap.a_grid, snap.a_pos)
        delay = (
            0
            if a_bar is None
            else max(0, int(round((a_bar - snap.a_pos) / max(snap.a_rate, 1e-6))))
        )

        with self._lock:
            self._state.armed_track_id = entry.track_id

        self._begin_transition_span(entry, snap, delay=delay, n_xfade=n_xfade, target=target)
        self.say(
            f"[transition] {entry.crossfade_bars} bars ({n_xfade / config.SR:.1f}s), "
            f"starting in about {delay / config.SR:.2f}s on the next bar"
            + (f" — {entry.plan_note}" if entry.plan_note else "")
        )

    def _autopilot(self, remaining: float) -> None:
        now = time.monotonic()
        if now - self._last_autopilot_at < AUTOPILOT_COOLDOWN_S:
            return
        self._last_autopilot_at = now
        message = (
            f"[autopilot] The queue is empty and there is about {remaining:.0f}s of music left. "
            f"Pick the next track around {self.target_bpm():.0f} BPM and queue it."
        )
        try:
            self.q_user.put_nowait(message)
        except Full:
            pass

    def _underrun(self, remaining: float) -> None:
        self._faded_out = True
        with self._lock:
            self._state.underruns += 1
        fade = bars_to_samples(4, self.target_bpm())
        self.mixer.post(FadeOut(gain_a=linear_fade_out(fade)))
        self.say(
            f"[underrun] nothing staged with {remaining:.1f}s left — fading out over 4 bars. "
            "The callback is never stalled; the set just ends."
        )
        if self._transition_span is not None:
            ins.set_attributes(self._transition_span, {"underrun": True})

    # ------------------------------------------------------------ transition spans

    def _begin_transition_span(
        self, entry: QueuedTrack, snap: Snapshot, *, delay: int, n_xfade: int, target: float
    ) -> None:
        self._end_transition_span(reason="superseded")
        tracer = ins.get_tracer()
        span = tracer.start_span("transition")
        ins.set_attributes(
            span,
            {
                "openinference.span.kind": ins.CHAIN,
                "from_track_id": snap.a_track_id,
                "to_track_id": entry.track_id,
                "crossfade_bars": entry.crossfade_bars,
                "crossfade_seconds": round(n_xfade / config.SR, 2),
                "bar_delay_ms_estimate": round(delay / config.SR * 1000.0, 1),
                "target_bpm": target,
                "cue_seconds": entry.cue_seconds,
                "plan_note": entry.plan_note or None,
                "underrun": False,
            },
        )
        self._transition_span = span

    def _end_transition_span(self, *, reason: str, snap: Snapshot | None = None) -> None:
        span = self._transition_span
        if span is None:
            return
        self._transition_span = None
        ins.set_attributes(span, {"outcome": reason})
        if snap is not None:
            ins.set_attributes(span, {"flips": snap.flips, "xruns": snap.xruns})
        ins.set_output(span, reason)
        span.end()

    # ------------------------------------------------------------ startup

    def prime(self, track: Track) -> bool:
        """Load and start the first track synchronously, before the stream opens.

        Done on the calling thread on purpose: the first librosa call pays a
        ~14s numba JIT warm-up, and paying it here means the set starts with
        audio instead of silence.
        """
        source = self.sources.get(track.source)
        if source is None:
            raise ToolError(f"source {track.source!r} is not enabled")
        with ins.traced_span(
            "track.prime", ins.CHAIN, input_value=track.label, attributes={"source": track.source}
        ) as span:
            record, audio, _ = library.fetch_and_analyze(track, source)
            grid = analysis_mod.bar_grid(record)
            with self._lock:
                self._state.target_bpm = record.bpm_used
                self._state.now_playing = NowPlaying(
                    track_id=track.track_id,
                    title=track.title,
                    artist=track.artist,
                    bpm_used=record.bpm_used,
                    started_at=time.monotonic(),
                )
            self.mixer.load_deck_a(
                LoadDeckB(
                    track_id=track.track_id,
                    buf=audio,
                    grid=grid,
                    rate=1.0,
                    start_pos=float(grid[0]) if grid.size else 0.0,
                )
            )
            self._ready_put(track.track_id, record, audio)
            library.append_credit(library.credit_for(track, record.bpm_used, None))
            ins.set_output(span, {"bpm": record.bpm_used, "grid_confidence": record.grid_confidence})
            self.say(
                f"[now playing] {track.label} — {record.bpm_used:.1f} BPM, "
                f"grid confidence {record.grid_confidence:.2f}"
                f"{library.license_flag(track.license_verified)}"
            )
        return True

    # ------------------------------------------------------------ tools

    def tool_get_now_playing(self) -> str:
        snap = self.snapshot()
        with self._lock:
            state = self._state
            np_ = state.now_playing
            queue = list(state.queue)
            target = state.target_bpm

        lines: list[str] = []
        if np_ is None:
            lines.append("Nothing is playing yet.")
        else:
            lines.append(
                f"Now playing: {np_.title} by {np_.artist} — {np_.bpm_used:.1f} BPM source, "
                f"playing at rate {snap.a_rate:.4f} ({np_.bpm_used * snap.a_rate:.1f} BPM out). "
                f"{snap.a_pos / config.SR:.0f}s in, {snap.a_remaining_out_s:.0f}s left."
            )
        lines.append(f"Target BPM: {target:.1f}")
        if snap.xfade_len:
            done = snap.xfade_pos / max(snap.xfade_len, 1) * 100
            lines.append(f"A crossfade is in progress, {done:.0f}% complete.")
        elif snap.b_staged:
            lines.append("Deck B is staged and ready.")
        else:
            lines.append("Deck B is empty.")
        if queue:
            lines.append(
                "Queue: "
                + ", ".join(
                    f"{q.track_id} ({q.crossfade_bars} bars)" for q in queue
                )
            )
        else:
            lines.append("Queue is empty.")
        lines.append(f"Transitions so far: {snap.flips}. Audio dropouts: {snap.xruns}.")
        if snap.callback_errors:
            lines.append(
                f"WARNING: the audio callback failed {snap.callback_errors} time(s) "
                f"(last: {snap.callback_error}). Some blocks were silent."
            )
        return "\n".join(lines)

    def tool_set_target_bpm(self, raw: dict) -> str:
        args = SetTargetBpmArgs.model_validate(raw)
        snap = self.snapshot()
        if snap.xfade_len:
            with self._lock:
                self._state.pending_target_bpm = args.bpm
            return (
                f"A crossfade is in progress, so both decks are already locked to the current "
                f"tempo. Deferred: the set will move to {args.bpm:.1f} BPM right after the "
                f"transition completes."
            )
        return self._apply_target_bpm(args.bpm, args.ramp_bars)

    def _apply_target_bpm(self, bpm: float, ramp_bars: int) -> str:
        snap = self.snapshot()
        with self._lock:
            np_ = self._state.now_playing
            old = self._state.target_bpm
        if np_ is None:
            with self._lock:
                self._state.target_bpm = bpm
            self._drop_unplayable(bpm)
            return f"Target BPM set to {bpm:.1f} (nothing playing yet)."

        wanted = bpm / np_.bpm_used
        rate, correction = clamp_rate(wanted)
        n = bars_to_samples(ramp_bars, old)
        self.mixer.post(SetRateRamp("a", rate_ramp(snap.a_rate, rate, n)))
        effective = np_.bpm_used * rate
        with self._lock:
            self._state.target_bpm = effective
        dropped = self._drop_unplayable(effective)
        note = f" {correction}." if correction else ""
        if dropped:
            note += (
                f" Dropped {len(dropped)} queued track(s) that no longer fit the "
                f"varispeed window at this tempo."
            )
        return (
            f"Ramping the current track ({np_.bpm_used:.1f} BPM source) to rate {rate:.4f} "
            f"= {effective:.1f} BPM over {ramp_bars} bars.{note}"
        )

    def _drop_unplayable(self, target: float) -> list[str]:
        """Remove queued tracks the new tempo puts outside the varispeed window.

        A track already staged on deck B, or already crossfading, is *kept*:
        dropping the queue entry would not stop it sounding, it would only make
        the session's state disagree with what the mixer is doing.

        Bumping the generation is what makes the rest stick: a load already in
        flight for a dropped track would otherwise finish and stage a track that
        can no longer be beatmatched. The loader checks `_is_stale` and discards it.
        """
        with self._lock:
            committed = {self._state.staged_track_id, self._state.armed_track_id} - {None}
            keep: list[QueuedTrack] = []
            dropped: list[str] = []
            kept_committed: list[str] = []
            for entry in self._state.queue:
                if bpm_within_varispeed(library.bpm_of(entry.track_id), target):
                    keep.append(entry)
                elif entry.track_id in committed:
                    keep.append(entry)
                    kept_committed.append(entry.track_id)
                else:
                    dropped.append(entry.track_id)
            if dropped:
                self._state.queue = keep

        # Outside the lock: no I/O while it is held.
        for track_id in kept_committed:
            self.say(
                f"[conductor] {track_id} no longer fits {target:.1f} BPM but is "
                "already on deck B; letting it play"
            )
        if not dropped:
            return []
        self._bump_generation()
        for track_id in dropped:
            self.say(f"[conductor] dropped {track_id}: too far from {target:.1f} BPM now")
        return dropped

    def tool_skip(self, raw: dict) -> str:
        args = SkipArgs.model_validate(raw)
        snap = self.snapshot()
        if snap.xfade_len:
            return "A crossfade is already running; let it finish."
        with self._lock:
            queue = list(self._state.queue)
        if not queue:
            return "Nothing is queued to skip to. Find and queue a track first."
        if not snap.b_staged:
            ready = self._ready_get(queue[0].track_id)
            if ready is None:
                return "Deck B is still loading — try again in a few seconds."
            self._stage(queue[0])
            return "Deck B was not staged yet; staging it now. Call skip again in a moment."

        entry = queue[0]
        entry.crossfade_bars = args.crossfade_bars
        self._arm(entry, snap)
        return (
            f"Skipping into {entry.track_id} with a {args.crossfade_bars}-bar crossfade "
            "starting on the next bar."
        )

    def tool_queue_track(self, raw: dict) -> str:
        args = QueueTrackArgs.model_validate(raw)
        with self._lock:
            if args.track_id in self._unloadable:
                return (
                    f"{args.track_id} could not be decoded earlier in this set, so it is "
                    "out of play. Pick a different track."
                )
            if self._state.ended:
                return (
                    "The set has already ended — deck A faded out and nothing is playing. "
                    "Queueing a track now would not start it; the run has to be restarted."
                )
        track = library.load_track(args.track_id)
        if track is None:
            return (
                f"No track with id {args.track_id!r} in the library. "
                "Search first, then queue one of the ids you got back."
            )
        bpm = library.effective_bpm(track)
        target = self.target_bpm()

        if bpm:
            rate = target / bpm
            if not bpm_within_varispeed(bpm, target):
                return (
                    f"{track.label} is {bpm:.1f} BPM, which needs rate {rate:.3f} to reach the "
                    f"{target:.1f} BPM target — outside the +/-8% varispeed window. "
                    "Pick something closer in tempo, or move the target first."
                )
        duration = library.duration_of(args.track_id) or track.duration_s
        if duration and duration > config.MAX_TRACK_SECONDS:
            return f"{track.label} is {duration / 60:.1f} minutes — too long for a set."

        plan = self.plan_for(args.track_id)
        entry = QueuedTrack(
            track_id=args.track_id,
            crossfade_bars=args.crossfade_bars,
            cue_seconds=args.cue_seconds,
            plan_note=plan.note if plan else "",
        )
        with self._lock:
            if any(q.track_id == args.track_id for q in self._state.queue):
                return f"{track.label} is already in the queue."
            if args.position == "next":
                self._state.queue.insert(0, entry)
            else:
                self._state.queue.append(entry)
            depth = len(self._state.queue)

        if not library.is_ready(args.track_id) or self._ready_get(args.track_id) is None:
            self._request_load(track, purpose="queue")

        snap = self.snapshot()
        eta = snap.a_remaining_out_s
        # bpm is None for an untagged track that has not been analysed yet, which
        # is the common case for a local crate -- analysis will compute it.
        tempo = f"{bpm:.1f} BPM" if bpm else "tempo not known yet"
        return (
            f"Queued {track.label} ({tempo}) with a {args.crossfade_bars}-bar crossfade. "
            f"Position {1 if args.position == 'next' else depth} of {depth}; "
            f"it should start in about {max(0.0, eta):.0f}s."
        )

    def remember_plan(self, track_id: str, plan: ValidatedPlan) -> None:
        with self._lock:
            self._plans[track_id] = plan

    def plan_for(self, track_id: str) -> ValidatedPlan | None:
        with self._lock:
            return self._plans.get(track_id)

    def tool_search_source(self, raw: dict) -> str:
        args = SearchSourceArgs.model_validate(raw)
        if args.source not in self.sources:
            enabled = ", ".join(sorted(self.sources)) or "none"
            return f"Source {args.source!r} is not enabled. Enabled sources: {enabled}."

        tags = tuple(t.strip() for t in (args.tags or "").split(",") if t.strip())
        query = SearchQuery(
            bpm_min=args.bpm_min, bpm_max=args.bpm_max, tags=tags, limit=args.limit
        )
        try:
            tracks = self.sources[args.source].search(query)
        except SourceError as exc:
            return f"Search failed on {args.source}: {exc}"

        if not tracks:
            return (
                f"No tracks on {args.source} between {args.bpm_min} and {args.bpm_max} BPM"
                + (f" tagged {args.tags}" if args.tags else "")
                + ". Try a wider tempo range or drop the tags."
            )

        for track in tracks:
            if library.load_track(track.track_id) is None:
                library.save_track(track)

        target = self.target_bpm()
        rows = [
            library.to_ref(track, target_bpm=target).model_dump(exclude_none=True)
            for track in tracks
        ]
        for row, track in zip(rows, tracks):
            if track.bpm and not bpm_within_varispeed(track.bpm, target):
                row["playable"] = False
                row["why_not"] = f"needs rate {target / track.bpm:.3f}, outside +/-8%"
            else:
                row["playable"] = True

        return json.dumps(rows, default=str)

    def tool_inspect_track(self, raw: dict) -> str:
        args = InspectTrackArgs.model_validate(raw)
        track = library.load_track(args.track_id)
        if track is None:
            return f"No track with id {args.track_id!r}."
        record = library.load_analysis(args.track_id)
        lines = [
            f"{track.label}",
            f"  source: {track.source}   license: {track.license_name} "
            f"(verified={track.license_verified})",
            f"  tags: {', '.join(track.tags) or '(none)'}",
        ]
        if record is None:
            lines.append(
                f"  not analysed yet. Metadata BPM: {track.bpm}. "
                "Queueing it will download and analyse it."
            )
        else:
            lines.append(
                f"  {record.bpm_used:.1f} BPM ({record.bpm_source}), "
                f"librosa estimated {record.bpm_librosa:.1f}, "
                f"grid confidence {record.grid_confidence:.2f} "
                f"({analysis_mod.grid_source(record)})"
            )
            lines.append(
                f"  {record.duration_s:.0f}s, {len(record.bar_indices)} bars, "
                f"loudness {record.rms_dbfs:.1f} dBFS"
            )
        return "\n".join(lines)

    def tool_get_analysis(self, raw: dict) -> str:
        args = GetAnalysisArgs.model_validate(raw)
        record = library.load_analysis(args.track_id)
        if record is None:
            return f"{args.track_id} has not been analysed yet."
        snap = self.snapshot()
        with self._lock:
            np_ = self._state.now_playing
        current = (
            f"Deck A: {np_.title} at {np_.bpm_used:.1f} BPM source, rate {snap.a_rate:.4f}, "
            f"{snap.a_remaining_out_s:.0f}s left."
            if np_
            else "Deck A: empty."
        )
        return (
            f"{current}\n"
            f"Deck B candidate {args.track_id}: {record.bpm_used:.1f} BPM "
            f"({record.bpm_source}), {record.duration_s:.0f}s, "
            f"{len(record.bar_indices)} bars, grid confidence {record.grid_confidence:.2f}.\n"
            f"Target BPM is {self.target_bpm():.1f}; playback rate would be "
            f"{self.target_bpm() / record.bpm_used:.4f} "
            f"(allowed window {config.RATE_MIN}-{config.RATE_MAX})."
        )

    # ------------------------------------------------------------ guardrail

    def validate_plan(self, raw: dict, to_track_id: str) -> ValidatedPlan:
        """Trace `schema.validate_plan` against the deck that is playing now.

        The clamping itself lives in `schema.py` beside the models it clamps;
        this owns the span and the one piece of session state it needs.
        """
        with ins.traced_span(
            "validate_plan", ins.GUARDRAIL, attributes={"to_track_id": to_track_id}
        ) as span:
            record = library.load_analysis(to_track_id)
            if record is None:
                raise ToolError(f"{to_track_id} has not been analysed; cannot plan a transition")
            with self._lock:
                np_ = self._state.now_playing

            validated = validate_plan(raw, record, np_.bpm_used if np_ else None)
            ins.set_attributes(
                span,
                {
                    "corrected": validated.was_corrected,
                    "corrections": validated.corrections,
                    "rate_a": validated.rate_a,
                    "rate_b": validated.rate_b,
                },
            )
            ins.set_output(span, validated.model_dump())
            return validated

    # ------------------------------------------------------------ reporting

    def report(self) -> str:
        snap = self.snapshot()
        with self._lock:
            state = self._state
            report = (
                f"{snap.flips} transitions, {state.underruns} underruns, "
                f"{snap.xruns} audio dropouts, {snap.frames_out / config.SR:.0f}s of audio, "
                f"target {state.target_bpm:.1f} BPM"
            )
        if snap.callback_errors:
            report += (
                f"\n{snap.callback_errors} audio callback failure(s) — last: {snap.callback_error}"
            )
        return report

    def find_candidates(
        self,
        bpm_min: float,
        bpm_max: float,
        *,
        tags: Sequence[str] = (),
        limit: int = 5,
    ) -> list[Track]:
        """Cache first, then each enabled source, first non-empty result wins.

        The cache comes first because an already-analysed track can go straight
        onto a deck with no download and no librosa pass. This is the only search
        in the project: the rule selector, the crate digger's fallback, and the
        opening-track pick all route through here.
        """
        with self._lock:
            unloadable = set(self._unloadable)

        cached = [
            t
            for t in library.search_cache(
                bpm_min, bpm_max, tags=tags, limit=limit, sources=self.sources.keys()
            )
            if t.track_id not in unloadable
        ]
        if cached:
            return cached

        for name, source in self.sources.items():
            try:
                hits = source.search(
                    SearchQuery(
                        bpm_min=bpm_min, bpm_max=bpm_max, tags=tuple(tags), limit=limit
                    )
                )
            except SourceError as exc:
                self.say(f"[{name}] {exc}")
                continue
            hits = [t for t in hits if t.track_id not in unloadable]
            if hits:
                # Persist metadata now so the ids we hand out are resolvable.
                for track in hits:
                    if library.load_track(track.track_id) is None:
                        library.save_track(track)
                return hits
        return []

    def spoken_for(self) -> set[str]:
        """Track ids already playing or queued, so a selector does not repeat one."""
        with self._lock:
            ids = {q.track_id for q in self._state.queue}
            if self._state.now_playing is not None:
                ids.add(self._state.now_playing.track_id)
        return ids

    def playable_window(self) -> tuple[float, float]:
        return playable_window(self.target_bpm())
