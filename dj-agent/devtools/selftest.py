"""Offline verification: the mixer's DSP, and cross-thread span propagation.

Generates two synthetic click tracks whose beat grids are known *exactly*, runs
them through the real `Mixer.process()` block loop, and asserts the four
properties that make a crossfade sound right:

  1. equal power   — the summed RMS never dips through the fade
  2. no clipping   — no sample exceeds full scale
  3. beat lock     — every deck-B click lands on a deck-A click, all 8 bars
  4. exact length  — the block loop emits precisely the frames requested

These run as `dj.py` flags rather than under pytest because both produce output
a human is meant to look at: `--selftest-mixer` writes an audible .wav so the
asserts can be confirmed by ear, and `--trace-selftest` prints the span tree it
checked. `tests/` reuses `click_track` from here rather than duplicating it.
"""

from __future__ import annotations

import numpy as np

from config import BEATS_PER_BAR, BLOCKSIZE, CHANNELS, OUT_DIR, SR
from audio.mixer import (
    LoadDeckB,
    Mixer,
    StartXfade,
    bars_to_samples,
    equal_power_curves,
    next_bar_after,
)

CLICK_MS = 6.0
N_PARTIALS = 48


def click_track(bpm: float, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (buf, bar_grid). One click per beat; downbeats accented.

    The grid is computed from the same integer beat positions the clicks are
    written at, so it is exact by construction — there is no analysis error to
    confuse a beat-lock failure with.
    """
    n = int(round(seconds * SR))
    buf = np.zeros((n, CHANNELS), dtype=np.float32)
    beat_len = SR * 60.0 / bpm
    click_n = int(round(CLICK_MS / 1000.0 * SR))
    env = np.exp(-np.linspace(0.0, 6.0, click_n)).astype(np.float32)
    tone = np.sin(2 * np.pi * 1800.0 * np.arange(click_n) / SR).astype(np.float32)
    click = (env * tone).astype(np.float32)

    beats: list[int] = []
    k = 0
    while True:
        pos = int(round(k * beat_len))
        if pos + click_n >= n:
            break
        beats.append(pos)
        gain = 0.6 * (1.0 if k % BEATS_PER_BAR == 0 else 0.55)
        buf[pos : pos + click_n, 0] += click * gain
        buf[pos : pos + click_n, 1] += click * gain
        k += 1

    grid = np.array([b for i, b in enumerate(beats) if i % BEATS_PER_BAR == 0], dtype=np.int64)
    return buf, grid


def onsets(mono: np.ndarray) -> np.ndarray:
    """Sample indices of click peaks.

    The threshold is relative to the loudest peak, not absolute, so this works
    on a fully-faded-in deck and a fading-out one alike.
    """
    env = np.abs(mono)
    peak = float(env.max())
    if peak <= 0:
        return np.zeros(0, dtype=np.int64)
    above = env > 0.25 * peak
    if not above.any():
        return np.zeros(0, dtype=np.int64)
    bounds = np.flatnonzero(np.diff(above.astype(np.int8)) != 0) + 1
    out = []
    for group in np.split(np.arange(len(above)), bounds):
        if len(group) and above[group[0]]:
            out.append(int(group[0] + int(np.argmax(env[group]))))
    return np.array(out, dtype=np.int64)


def _bandlimited_probe(n: int, seed: int) -> np.ndarray:
    """Decorrelated, band-limited test signal with a flat envelope.

    Used for the equal-power assertion, where the probe must survive varispeed
    resampling without losing energy. Frequencies are capped at 2 kHz, far
    enough below Nyquist that linear interpolation is transparent.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / SR
    freqs = rng.uniform(120.0, 2000.0, N_PARTIALS)
    phases = rng.uniform(0.0, 2 * np.pi, N_PARTIALS)
    sig = np.zeros(n, dtype=np.float64)
    for f, ph in zip(freqs, phases):
        sig += np.sin(2 * np.pi * f * t + ph)
    sig *= 0.25 / np.sqrt((sig**2).mean())
    mono = sig.astype(np.float32)
    return np.stack([mono, mono], axis=1)


def _schedule(
    mixer: Mixer,
    buf_b: np.ndarray,
    grid_b: np.ndarray,
    xfade_bars: int,
    target_bpm: float,
    rate_b: float,
    *,
    flat_gains: bool = False,
) -> int:
    """Stage deck B and arm a beat-aligned crossfade. Returns delay_frames.

    `flat_gains` renders both decks at unity instead of crossfading, which
    isolates *timing* from *level* — the beat-lock assertion uses it so a
    faded-out click can never be mistaken for a mistimed one.
    """
    a_bar = next_bar_after(mixer.a.grid, mixer.a.pos)
    if a_bar is None:
        raise RuntimeError("deck A has no bar boundary ahead of the playhead")
    delay_frames = int(round((a_bar - mixer.a.pos) / mixer.a.rate))

    # Deck B enters on one of its own bar boundaries, so the two downbeats
    # coincide at output-frame delay_frames.
    b_start = int(grid_b[1]) if grid_b.size > 1 else 0

    mixer.post(LoadDeckB(track_id="B", buf=buf_b, grid=grid_b, rate=rate_b, start_pos=b_start))
    n_xfade = bars_to_samples(xfade_bars, target_bpm)
    if flat_gains:
        gain_a = np.ones(n_xfade, dtype=np.float32)
        gain_b = np.ones(n_xfade, dtype=np.float32)
    else:
        gain_a, gain_b = equal_power_curves(n_xfade)
    mixer.post(StartXfade(delay_frames=delay_frames, gain_a=gain_a, gain_b=gain_b))
    return delay_frames


def _pump(mixer: Mixer, total_frames: int) -> np.ndarray:
    """Drive process() the way PortAudio would, but offline and deterministic."""
    chunks = []
    done = 0
    while done < total_frames:
        n = min(BLOCKSIZE, total_frames - done)
        chunks.append(mixer.process(n).copy())
        done += n
    return np.concatenate(chunks, axis=0)


def run(
    bpm_a: float = 120.0,
    bpm_b: float = 128.0,
    seconds: float = 30.0,
    xfade_bars: int = 8,
    write_wav: bool = True,
) -> int:
    target_bpm = bpm_a
    rate_b = bpm_a / bpm_b  # varispeed deck B down to deck A's tempo
    print(f"deck A {bpm_a:g} BPM @ rate 1.0000 | deck B {bpm_b:g} BPM @ rate {rate_b:.4f}")

    buf_a, grid_a = click_track(bpm_a, seconds)
    buf_b, grid_b = click_track(bpm_b, seconds)

    n_xfade = bars_to_samples(xfade_bars, target_bpm)
    total = int(n_xfade + 6.0 * SR)

    def render(
        a: np.ndarray,
        b: np.ndarray,
        grid_for_a: np.ndarray,
        grid_for_b: np.ndarray,
        *,
        flat_gains: bool = False,
    ) -> tuple[np.ndarray, int, Mixer]:
        m = Mixer()
        m.load_deck_a(
            LoadDeckB(track_id="A", buf=a, grid=grid_for_a, rate=1.0, start_pos=int(grid_for_a[1]))
        )
        delay = _schedule(
            m, b, grid_for_b, xfade_bars, target_bpm, rate_b, flat_gains=flat_gains
        )
        return _pump(m, total), delay, m

    # --- the audible render: click tracks through the real equal-power fade
    mix, delay_frames, m_mix = render(buf_a, buf_b, grid_a, grid_b)

    if write_wav:
        import soundfile as sf

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "selftest.wav"
        sf.write(path, mix, SR)
        print(f"wrote {path}  ({len(mix) / SR:.1f}s)")

    failures: list[str] = []
    xf0, xf1 = delay_frames, delay_frames + n_xfade

    # ---- 1. exact length
    if len(mix) != total:
        failures.append(f"length: got {len(mix)} frames, expected {total}")
    if m_mix.snapshot.frames_out > total:
        failures.append(f"frames_out overshoot: {m_mix.snapshot.frames_out} > {total}")

    # ---- 2. no clipping
    peak = float(np.abs(mix).max())
    if peak > 1.0 + 1e-6:
        failures.append(f"clipping: peak {peak:.4f} > 1.0")

    # ---- 3. equal power, measured on DECORRELATED material.
    # Click tracks are the wrong probe here: once beatmatched, the two decks are
    # perfectly correlated, so cos/sin gains sum to +3 dB at the midpoint by
    # simple amplitude addition. That is correct physics, not a mixer fault.
    # Real tracks are uncorrelated, so noise is the honest probe.
    # The probe must also be BAND-LIMITED. Linear interpolation at a rate
    # below 1.0 lowpasses its input, which costs white noise ~1.8 dB of RMS --
    # so white noise would read as a 1.8 dB "dip" that is really the resampler
    # doing its job. Random-phase sinusoids well below Nyquist interpolate
    # transparently and are still mutually decorrelated.
    probe_a = _bandlimited_probe(len(buf_a), seed=0xD1)
    probe_b = _bandlimited_probe(len(buf_b), seed=0x5E)
    noise_mix, n_delay, _ = render(probe_a, probe_b, grid_a, grid_b)
    win = SR // 4
    rms = [
        float(np.sqrt((noise_mix[s : s + win].astype(np.float64) ** 2).mean()))
        for s in range(n_delay, n_delay + n_xfade - win, win)
    ]
    rms_db = 20 * np.log10(np.array(rms) + 1e-12)
    dip = float(rms_db.max() - rms_db.min())
    if dip > 1.5:
        failures.append(f"equal power: RMS varies {dip:.2f} dB across the fade (> 1.5 dB)")

    # ---- 4. beat lock across the whole fade, at flat gain so a quiet click
    # can never be misread as a late one.
    a_only, _, _ = render(buf_a, np.zeros_like(buf_b), grid_a, grid_b, flat_gains=True)
    b_only, _, _ = render(np.zeros_like(buf_a), buf_b, grid_a, grid_b, flat_gains=True)
    a_on = onsets(a_only[:, 0])
    b_on = onsets(b_only[:, 0])
    b_in = b_on[(b_on >= xf0) & (b_on < xf1)]
    a_in = a_on[(a_on >= xf0 - SR) & (a_on < xf1 + SR)]
    expected_clicks = BEATS_PER_BAR * xfade_bars
    worst_ms = float("nan")
    if len(b_in) < expected_clicks - 2:
        failures.append(
            f"beat lock: only {len(b_in)} deck-B clicks inside the fade "
            f"(expected ~{expected_clicks})"
        )
    elif len(a_in) == 0:
        failures.append("beat lock: no deck-A clicks near the fade")
    else:
        nearest = np.abs(b_in[:, None] - a_in[None, :]).min(axis=1)
        worst_ms = float(nearest.max()) / SR * 1000.0
        median_ms = float(np.median(nearest)) / SR * 1000.0
        if worst_ms > 2.0:
            failures.append(f"beat lock: worst deck-B click is {worst_ms:.2f} ms off (> 2.0 ms)")
        print(
            f"beat lock: {len(b_in)} deck-B clicks, worst offset {worst_ms:.3f} ms, "
            f"median {median_ms:.3f} ms"
        )

    print(f"peak {peak:.4f} | RMS spread across fade {dip:.2f} dB | flips {m_mix.snapshot.flips}")
    print(f"xfade window: frames {xf0}..{xf1} ({n_xfade / SR:.2f}s at {target_bpm:g} BPM)")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK — all four mixer assertions passed")
    return 0


# ---------------------------------------------------------------- tracing


def run_tracing() -> int:
    """Assert that OTel context survives a thread hand-off, in-process.

    The span names below are *mocks* -- a stand-in tree the shape of a real turn,
    not the production spans. What is being verified is the one thing that is
    easy to get silently wrong and invisible in a passing test suite: spans
    started on a pool thread must be children of the span that dispatched them.
    The negative control at the end runs the same worker *without*
    `propagate_to_thread` and confirms the span really does become an orphan
    root, which is what proves the helper is load-bearing and not decoration.
    """
    from concurrent.futures import ThreadPoolExecutor

    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import instrumentation as ins

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    ins.use_provider(provider)

    import threading

    threads: dict[str, str] = {}

    def digger(ctx) -> None:  # runs on a pool thread
        with ins.propagate_to_thread(ctx):
            with ins.traced_span(
                "mock.sub_agent", ins.AGENT, attributes={"role": "crate_digger", "model": "cheap"}
            ) as span:
                threads["mock.sub_agent"] = threading.current_thread().name
                with ins.traced_span("mock.sub_agent.step_one", ins.TOOL, attributes={"source": "local"}):
                    pass
                with ins.traced_span("mock.sub_agent.step_two", ins.TOOL):
                    pass
                ins.set_output(span, {"candidates": 2})

    def loader(ctx) -> None:  # runs on the "loader thread"
        with ins.propagate_to_thread(ctx):
            with ins.traced_span("mock.loader", ins.CHAIN, attributes={"track_id": "abc123"}):
                threads["mock.loader"] = threading.current_thread().name
                with ins.traced_span("mock.loader.fetch", ins.TOOL, attributes={"cache_hit": True}):
                    pass

    def orphan(_ctx) -> None:  # negative control: no propagation
        with ins.traced_span("mock.orphan_no_propagation", ins.TOOL):
            pass

    with ins.using_set_context("set-selftest"):
        with ins.traced_span("mock.turn", ins.CHAIN, input_value="more upbeat") as turn:
            ctx = ins.capture_context()
            with ins.traced_span(
                "mock.host", ins.AGENT, attributes={"role": "host", "model": "expensive"}
            ):
                threads["mock.host"] = threading.current_thread().name
                with ins.traced_span("mock.host.step_one", ins.TOOL):
                    pass
                with ins.traced_span("mock.host.delegate", ins.TOOL):
                    find_ctx = ins.capture_context()
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        pool.submit(digger, find_ctx).result()
                with ins.traced_span("mock.host.step_two", ins.TOOL, attributes={"crossfade_bars": 8}):
                    pass
            with ThreadPoolExecutor(max_workers=2) as pool:
                pool.submit(loader, ctx).result()
                pool.submit(orphan, ctx).result()
            ins.set_output(turn, "queued 'Solar Drift'")

    provider.force_flush()
    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    ids = {s.context.span_id: s.name for s in spans}

    def parent_of(name: str) -> str | None:
        span = by_name.get(name)
        if span is None or span.parent is None:
            return None
        return ids.get(span.parent.span_id)

    failures: list[str] = []

    expected_parents = {
        "mock.turn": None,
        "mock.host": "mock.turn",
        "mock.host.step_one": "mock.host",
        "mock.host.delegate": "mock.host",
        "mock.sub_agent": "mock.host.delegate",  # across a pool thread
        "mock.sub_agent.step_one": "mock.sub_agent",
        "mock.sub_agent.step_two": "mock.sub_agent",
        "mock.host.step_two": "mock.host",
        "mock.loader": "mock.turn",  # across the loader thread
        "mock.loader.fetch": "mock.loader",
        "mock.orphan_no_propagation": None,  # the negative control
    }
    for name, want in expected_parents.items():
        if name not in by_name:
            failures.append(f"missing span {name!r}")
            continue
        got = parent_of(name)
        if got != want:
            failures.append(f"{name!r} parent is {got!r}, expected {want!r}")

    # Cross-thread spans must really have run on other threads, or the test
    # would pass trivially.
    if threads.get("mock.sub_agent") == threads.get("mock.host"):
        failures.append("the sub-agent ran on the host thread; the test proves nothing")
    if threads.get("mock.loader") == threads.get("mock.host"):
        failures.append("the loader ran on the host thread; the test proves nothing")

    roots = sorted(s.name for s in spans if s.parent is None)
    if roots != ["mock.orphan_no_propagation", "mock.turn"]:
        failures.append(f"unexpected trace roots: {roots}")

    trace_ids = {s.context.trace_id for s in spans if s.name != "mock.orphan_no_propagation"}
    if len(trace_ids) != 1:
        failures.append(f"the propagated spans span {len(trace_ids)} traces, expected 1")

    missing_kind = [s.name for s in spans if "openinference.span.kind" not in (s.attributes or {})]
    if missing_kind:
        failures.append(f"spans without openinference.span.kind: {missing_kind}")

    sessioned = [
        s.name for s in spans if (s.attributes or {}).get("session.id") == "set-selftest"
    ]

    print(f"{len(spans)} spans, {len(trace_ids)} trace id for the propagated tree")
    print(f"roots: {roots}")
    print(
        f"threads: host={threads.get('mock.host')} "
        f"sub_agent={threads.get('mock.sub_agent')} loader={threads.get('mock.loader')}"
    )
    print(f"spans carrying session.id=set-selftest: {len(sessioned)}/{len(spans)}")
    print("tree:")
    for name in expected_parents:
        if name in by_name:
            print(f"  {name:26s} parent={parent_of(name)}")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "\nOK — OTel context survives both thread hand-offs, and the negative "
        "control orphans without it"
    )
    return 0
