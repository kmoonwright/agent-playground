"""Two-deck realtime mixer: varispeed playback and sample-accurate crossfades.

THE CALLBACK RULE
=================
`Mixer.process()` runs on the PortAudio callback thread. It must never acquire a
lock, touch the filesystem or network, call librosa, emit an OpenTelemetry span,
log, or allocate unboundedly. Nothing outside the callback ever mutates a `Deck`.
Communication is one-way in each direction: commands in through `post()` (a
bounded deque), state out through `snapshot` (a single atomic attribute rebind),
and finished buffers out through `retired` so the conductor thread pays for the
free instead of the audio thread.

This module deliberately imports nothing from `instrumentation` — see the rule
above. It also has no concept of tracks, sources, or BPM as such: it takes
buffers, rates, and precomputed gain curves, and it mixes them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config import BEATS_PER_BAR, BLOCKSIZE, CHANNELS, SR

# ---------------------------------------------------------------- commands
# Tiny frozen dataclasses. Everything expensive (decoded audio, gain curves)
# is computed on the conductor or loader thread and shipped in ready to use.


@dataclass(frozen=True)
class LoadDeckB:
    """Stage a track on deck B without starting it."""

    track_id: str
    buf: np.ndarray  # (n, 2) float32 at SR, already loudness-normalized
    grid: np.ndarray  # bar-boundary sample indices into buf
    rate: float
    start_pos: float  # sample offset in buf where the fade-in should begin


@dataclass(frozen=True)
class StartXfade:
    """Begin a crossfade after `delay_frames` output frames.

    `gain_a` / `gain_b` are precomputed float32 arrays of equal length; the
    callback only slices them, so it performs no trig and no allocation.

    With `align_to_bar`, the callback computes the delay itself from deck A's
    grid and position at the moment the command is applied, and `delay_frames`
    is ignored. That is the only way to be sample-accurate: the producer's view
    of deck A comes from a snapshot published every 8 blocks, so a delay it
    computed would be up to ~186 ms stale by the time the callback saw it.
    """

    delay_frames: int
    gain_a: np.ndarray
    gain_b: np.ndarray
    align_to_bar: bool = False


@dataclass(frozen=True)
class SetRateRamp:
    """Ramp a deck's playback rate over `len(rates)` output frames."""

    deck: str  # "a" or "b"
    rates: np.ndarray  # float32, per-frame target rates


@dataclass(frozen=True)
class FadeOut:
    """Fade deck A to silence (underrun rescue)."""

    gain_a: np.ndarray


@dataclass(frozen=True)
class NudgeDeck:
    """Shift a deck's playhead by whole samples.

    Exists because there is no dependable downbeat tracker: when the bar grid is
    phased a beat off, the fix is a human ear and a small offset.
    """

    deck: str  # "a" or "b"
    frames: int


@dataclass(frozen=True)
class Stop:
    pass


Command = LoadDeckB | StartXfade | SetRateRamp | FadeOut | NudgeDeck | Stop


# ---------------------------------------------------------------- state


@dataclass
class Deck:
    """Callback-owned. Never read or written from another thread."""

    track_id: str | None = None
    buf: np.ndarray | None = None
    grid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    pos: float = 0.0
    rate: float = 1.0
    active: bool = False
    ramp: np.ndarray | None = None
    ramp_pos: int = 0

    @property
    def n(self) -> int:
        return 0 if self.buf is None else self.buf.shape[0]


@dataclass(frozen=True)
class Snapshot:
    """Immutable view published by the callback for every other thread."""

    a_track_id: str | None = None
    a_pos: float = 0.0
    a_len: int = 0
    a_rate: float = 1.0
    a_active: bool = False
    # Deck A's bar grid, so the conductor can align a cue point without reading
    # the callback-owned Deck. compare=False because ndarray == ndarray is not a
    # bool; nothing compares Snapshots, but the frozen dataclass would try.
    a_grid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), compare=False)
    b_staged: bool = False
    xfade_pos: int = 0
    xfade_len: int = 0
    flips: int = 0
    xruns: int = 0
    frames_out: int = 0
    # Counted so a failure on the audio thread is visible instead of silent.
    callback_errors: int = 0
    callback_error: str | None = None
    dropped_xfades: int = 0

    @property
    def a_remaining_out_s(self) -> float:
        """Output seconds left on deck A at its current rate."""
        if not self.a_active or self.a_len == 0 or self.a_rate <= 0:
            return 0.0
        return max(0.0, (self.a_len - self.a_pos) / self.a_rate / SR)


SNAPSHOT_EVERY_BLOCKS = 8
FLIP_LOG_LIMIT = 256


class Mixer:
    """Deck A + deck B, varispeed playback, equal-power crossfade."""

    def __init__(self, blocksize: int = BLOCKSIZE) -> None:
        self.a = Deck()
        self.b = Deck()

        # T3/T5 -> T4. deque.append/popleft are atomic under the GIL, so the
        # callback needs no lock. maxlen bounds it so a runaway producer cannot
        # make the callback's drain loop unbounded.
        self.cmds: deque[Command] = deque(maxlen=32)
        # T4 -> T5. Dropping an ~80 MB array's last reference is a free() the
        # audio thread should not pay for.
        self.retired: deque[np.ndarray] = deque()

        self.snapshot = Snapshot()
        self._blocks = 0
        self._flips = 0
        self._xruns = 0
        self._frames_out = 0
        self._stopped = False
        self._callback_errors = 0
        self._callback_error: str | None = None
        self._dropped_xfades = 0

        self._xfade_gain_a: np.ndarray | None = None
        self._xfade_gain_b: np.ndarray | None = None
        self._xfade_pos = 0
        self._xfade_len = 0
        self._pending_delay = 0
        self._pending_xfade: StartXfade | None = None

        self._fade_gain: np.ndarray | None = None
        self._fade_pos = 0

        # The one buffer worth reusing: the output block, which the callback
        # hands straight to PortAudio. See the aliasing note on process().
        self._scratch = np.zeros((blocksize, CHANNELS), dtype=np.float32)

        # Read by the offline selftest and the renderer, but appended to on the
        # callback thread, so it is capped rather than left to grow all set.
        self.flip_log: list[tuple[int, str]] = []

    # ------------------------------------------------------------ producer API

    def post(self, cmd: Command) -> None:
        """Called from T3/T5. Never from the callback."""
        self.cmds.append(cmd)

    def load_deck_a(self, cmd: LoadDeckB) -> None:
        """Prime deck A synchronously before the stream opens (launch only)."""
        self.a = Deck(
            track_id=cmd.track_id,
            buf=cmd.buf,
            grid=cmd.grid,
            pos=float(cmd.start_pos),
            rate=float(cmd.rate),
            active=True,
        )
        self._publish()

    def drain_retired(self) -> int:
        """Called from T5. Returns how many buffers were released."""
        released = 0
        while True:
            try:
                self.retired.popleft()
            except IndexError:
                break
            released += 1
        return released

    # ------------------------------------------------------------ callback

    def sd_callback(self, outdata, frames, time_info, status) -> None:
        """sounddevice OutputStream callback.

        A truthy `status` is PortAudio reporting an underflow or overflow -- the
        audible kind of failure -- so it is counted here and surfaced in the
        snapshot rather than swallowed.

        Nothing may propagate out of here: an exception aborts the PortAudio
        stream, which would leave the set permanently silent while the REPL kept
        accepting commands. One block of silence plus a counter the conductor and
        `/status` can see is the survivable failure.
        """
        if status:
            self._xruns += 1
        try:
            outdata[:] = self.process(frames)
        except Exception as exc:
            outdata[:] = 0.0
            self._callback_errors += 1
            self._callback_error = f"{type(exc).__name__}: {exc}"
            self._publish()

    def process(self, frames: int) -> np.ndarray:
        """Render `frames` output frames. Pure realtime path; also used offline.

        Returns a VIEW into a reusable output buffer. `sd_callback` copies it
        out immediately via `outdata[:] = ...`; any offline caller that keeps
        the result across calls must `.copy()` it, or every retained block will
        alias the last one.

        The buffer is sized from the blocksize passed to `__init__`, which is
        the same blocksize the stream is opened with, so a larger `frames` means
        the two have drifted apart -- hence the assert rather than a silent
        reallocation on the audio thread.
        """
        assert frames <= self._scratch.shape[0], (
            f"callback asked for {frames} frames but the mixer was built for "
            f"{self._scratch.shape[0]}"
        )
        out = self._scratch[:frames]
        out[:] = 0.0

        self._drain_commands()

        written = 0
        while written < frames:
            seg = frames - written
            if self._pending_delay > 0:
                seg = min(seg, self._pending_delay)
            if self._xfade_len:
                # Not `elif`: a delay and a running crossfade can both be live,
                # and dropping this bound overruns the gain arrays.
                seg = min(seg, self._xfade_len - self._xfade_pos)
            if self._fade_gain is not None:
                seg = min(seg, len(self._fade_gain) - self._fade_pos)
            # Every clause above is bounded below by 1: a pending delay is
            # positive, an active crossfade has not reached its end, and a
            # finished fade is cleared in the same block it completes.
            assert seg > 0, "zero-length segment would spin the render loop"
            self._render(out[written : written + seg], seg)
            written += seg

        self._frames_out += frames
        self._blocks += 1
        if self._blocks % SNAPSHOT_EVERY_BLOCKS == 0:
            self._publish()
        return out

    # ------------------------------------------------------------ internals

    def _drain_commands(self) -> None:
        # Bounded by cmds.maxlen, so this loop cannot run long.
        for _ in range(self.cmds.maxlen):
            try:
                cmd = self.cmds.popleft()
            except IndexError:
                return
            self._apply(cmd)

    def _apply(self, cmd: Command) -> None:
        if isinstance(cmd, LoadDeckB):
            if self.b.buf is not None:
                self.retired.append(self.b.buf)
            self.b = Deck(
                track_id=cmd.track_id,
                buf=cmd.buf,
                grid=cmd.grid,
                pos=float(cmd.start_pos),
                rate=float(cmd.rate),
                active=False,
            )
        elif isinstance(cmd, StartXfade):
            if self.b.buf is None or self._xfade_len > 0 or self._pending_xfade is not None:
                # Nothing staged, or a crossfade is already running or armed.
                # Counted so the conductor can re-post rather than wait forever.
                self._dropped_xfades += 1
                return
            self._pending_xfade = cmd
            self._pending_delay = max(0, self._delay_for(cmd))
            if self._pending_delay == 0:
                self._begin_xfade()
        elif isinstance(cmd, SetRateRamp):
            deck = self._deck(cmd.deck)
            deck.ramp = cmd.rates
            deck.ramp_pos = 0
        elif isinstance(cmd, NudgeDeck):
            deck = self._deck(cmd.deck)
            if deck.buf is not None:
                deck.pos = max(0.0, min(float(deck.n - 2), deck.pos + cmd.frames))
        elif isinstance(cmd, FadeOut):
            self._fade_gain = cmd.gain_a
            self._fade_pos = 0
        elif isinstance(cmd, Stop):
            self._stopped = True
            self.a.active = False
            self.b.active = False

    def _delay_for(self, cmd: StartXfade) -> int:
        """Frames until the crossfade should begin. Callback thread only."""
        if not cmd.align_to_bar:
            return int(cmd.delay_frames)
        bar = next_bar_after(self.a.grid, self.a.pos)
        if bar is None:
            return 0
        return int(round((bar - self.a.pos) / max(self.a.rate, 1e-6)))

    def _deck(self, name: str) -> Deck:
        return self.a if name == "a" else self.b

    def _begin_xfade(self) -> None:
        cmd = self._pending_xfade
        if cmd is None:
            return
        self._xfade_gain_a = cmd.gain_a
        self._xfade_gain_b = cmd.gain_b
        self._xfade_len = int(min(len(cmd.gain_a), len(cmd.gain_b)))
        self._xfade_pos = 0
        self._pending_xfade = None
        self.b.active = True

    def _render(self, dst: np.ndarray, seg: int) -> None:
        if self._stopped:
            return

        in_xfade = self._xfade_len > 0
        if in_xfade:
            sl = slice(self._xfade_pos, self._xfade_pos + seg)
            ga = self._xfade_gain_a[sl][:, None]
            gb = self._xfade_gain_b[sl][:, None]
        elif self._fade_gain is not None:
            sl = slice(self._fade_pos, self._fade_pos + seg)
            ga = self._fade_gain[sl][:, None]
            gb = None
        else:
            ga = None
            gb = None

        if self.a.active and self.a.buf is not None:
            block = self._read(self.a, seg)
            if ga is None:
                dst += block
            else:
                dst += block * ga

        if in_xfade and self.b.active and self.b.buf is not None:
            block = self._read(self.b, seg)
            dst += block * gb

        np.clip(dst, -1.0, 1.0, out=dst)

        if self._pending_delay > 0:
            self._pending_delay -= seg
            if self._pending_delay <= 0:
                self._pending_delay = 0
                self._begin_xfade()

        if in_xfade:
            self._xfade_pos += seg
            if self._xfade_pos >= self._xfade_len:
                self._flip()

        if self._fade_gain is not None and not in_xfade:
            self._fade_pos += seg
            if self._fade_pos >= len(self._fade_gain):
                self._fade_gain = None
                self._fade_pos = 0
                self.a.active = False

    def _read(self, deck: Deck, seg: int) -> np.ndarray:
        """Varispeed read with linear interpolation. Zero-pads past end of buffer.

        This is the whole of beatmatching: playing a buffer at a fractional rate
        shifts its tempo (and pitch) exactly the way a turntable pitch fader does.
        """
        buf = deck.buf
        assert buf is not None

        if deck.ramp is not None:
            take = deck.ramp[deck.ramp_pos : deck.ramp_pos + seg]
            if len(take):
                deck.rate = float(take[-1])
            deck.ramp_pos += seg
            if deck.ramp_pos >= len(deck.ramp):
                deck.ramp = None
                deck.ramp_pos = 0

        idx = deck.pos + deck.rate * np.arange(seg, dtype=np.float64)
        # idx increases monotonically, so searchsorted gives the count of leading
        # in-bounds samples. -2 leaves room for the i0+1 interpolation tap.
        n = min(int(np.searchsorted(idx, deck.n - 2, side="right")), seg) if deck.n > 2 else 0

        out = np.zeros((seg, CHANNELS), dtype=np.float32)
        if n:
            i0 = idx[:n].astype(np.int64)  # truncation == floor, since pos >= 0
            frac = (idx[:n] - i0).astype(np.float32)[:, None]
            out[:n] = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac

        deck.pos += deck.rate * seg
        return out

    def _flip(self) -> None:
        """Deck B becomes deck A. The old A buffer goes to the retire queue."""
        if self.a.buf is not None:
            self.retired.append(self.a.buf)
        self.a = self.b
        self.a.active = True
        self.b = Deck()
        self._xfade_gain_a = None
        self._xfade_gain_b = None
        self._xfade_len = 0
        self._xfade_pos = 0
        self._flips += 1
        if len(self.flip_log) < FLIP_LOG_LIMIT:
            self.flip_log.append((self._frames_out, self.a.track_id or "?"))

    def _publish(self) -> None:
        self.snapshot = Snapshot(
            a_track_id=self.a.track_id,
            a_pos=self.a.pos,
            a_len=self.a.n,
            a_rate=self.a.rate,
            a_active=self.a.active,
            a_grid=self.a.grid,
            b_staged=self.b.buf is not None,
            xfade_pos=self._xfade_pos,
            xfade_len=self._xfade_len,
            flips=self._flips,
            xruns=self._xruns,
            frames_out=self._frames_out,
            callback_errors=self._callback_errors,
            callback_error=self._callback_error,
            dropped_xfades=self._dropped_xfades,
        )


# ---------------------------------------------------------------- helpers
# Pure functions. Called from T5, never the callback, so the callback only ever
# slices the arrays they return rather than computing curves inline.


def bars_to_samples(bars: float, bpm: float) -> int:
    """Length of `bars` bars at `bpm`, in samples of the output clock."""
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    return int(round(bars * BEATS_PER_BAR * (60.0 / bpm) * SR))


def equal_power_curves(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Constant-power crossfade: gA=cos(t*pi/2), gB=sin(t*pi/2), gA^2+gB^2=1."""
    if n <= 0:
        raise ValueError("crossfade length must be positive")
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.cos(t * np.pi / 2).astype(np.float32), np.sin(t * np.pi / 2).astype(np.float32)


def linear_fade_out(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("fade length must be positive")
    return np.linspace(1.0, 0.0, n, dtype=np.float32)


def rate_ramp(from_rate: float, to_rate: float, n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("ramp length must be positive")
    return np.linspace(from_rate, to_rate, n, dtype=np.float32)


def next_bar_after(grid: np.ndarray, pos: float) -> int | None:
    """Smallest bar-boundary sample index strictly greater than `pos`."""
    if grid.size == 0:
        return None
    i = int(np.searchsorted(grid, pos, side="right"))
    if i >= grid.size:
        return None
    return int(grid[i])
