"""dj-agent entrypoint: argument parsing, the REPL, and the console printer.

This is the only module that reads argv, writes to the terminal, or opens a
PortAudio stream. Everything below it takes a `say(...)` callable instead of
printing, so no library module depends on the REPL. The selftests live in
`devtools/` and are reached through the flags parsed here.
"""

from __future__ import annotations

import argparse
import queue
import readline  # noqa: F401  -- importing it is what enables line editing
import sys
import threading
import time
import uuid
from pathlib import Path

import config

_print_lock = threading.Lock()
_prompt = "dj> "
_prompt_active = False
_is_tty = sys.stdout.isatty()


def say(message: str) -> None:
    """Print from any thread without eating a half-typed input line.

    The loader, the conductor, and the brain all speak through here while the
    user may be mid-word at the prompt, so this is also the one chokepoint where
    untrusted text is sanitized.
    """
    message = config.sanitize(message)
    with _print_lock:
        redraw = _prompt_active and _is_tty
        if redraw:
            sys.stdout.write("\r\x1b[2K")
        sys.stdout.write(message.rstrip("\n") + "\n")
        if redraw:
            sys.stdout.write(_prompt + readline.get_line_buffer())
        sys.stdout.flush()


def _read_line() -> str | None:
    global _prompt_active
    _prompt_active = True
    try:
        return input(_prompt)
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        _prompt_active = False


SLASH_HELP = """commands:
  <anything else>      talk to the DJ ("more upbeat", "drop to 100 bpm", "something with vocals")
  /status              deck state, queue, target BPM
  /credits             what has played so far, with licenses
  /nudge <±ms>         shift deck B by milliseconds (fixes a mis-phased bar grid by ear)
  /help                this list
  /quit                end the set and write out/SETLIST.md"""


# ---------------------------------------------------------------- shared setup


def _build_session(args: argparse.Namespace, prefix: str):
    """Sources, mixer, and session. Shared by the live set and cache seeding.

    Returns `(session, mixer, provider, set_id)`, or `None` if no source is
    usable — the caller turns that into an exit code.
    """
    import instrumentation
    from audio.mixer import Mixer
    from session import DJSession
    from music.source import SourceError
    from music.registry import build_sources, default_spec

    config.ensure_dirs()
    provider = config.require_provider(args.provider)

    # Tracing must be initialized before any LLM SDK import so the
    # instrumentors can patch them (see instrumentation.py).
    instrumentation.ensure_tracing(log_to_console=args.trace_console)

    try:
        sources = build_sources(
            args.source or default_spec(args.offline),
            crate_dir=args.crate,
        )
    except SourceError as exc:
        print(f"error: {exc}")
        return None

    set_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    mixer = Mixer(blocksize=args.blocksize)
    session = DJSession(mixer=mixer, sources=sources, say=say, set_id=set_id)
    return session, mixer, provider, set_id


def _prime_any(session, candidates, instrumentation, set_id: str) -> bool:
    """Try candidates in order until one actually loads.

    A first track can fail for real reasons -- a download that 403s, a file that
    will not decode -- and giving up on the first failure means no set at all.
    """
    with instrumentation.using_set_context(set_id):
        for track in candidates:
            try:
                session.prime(track)
                return True
            except Exception as exc:
                say(f"could not start with {track.label}: {type(exc).__name__}: {exc}")
    say("None of the candidate tracks could be loaded.")
    return False


# ---------------------------------------------------------------- the set


def run_set(args: argparse.Namespace) -> int:
    """Play a set: either live through PortAudio, or rendered to a wav file."""
    import agents.brain as brain_mod
    import instrumentation
    import music.library as library
    from schema import playable_window

    built = _build_session(args, "render" if args.render else "set")
    if built is None:
        return 2
    session, mixer, provider, set_id = built

    say(f"dj-agent — {set_id}")
    say(f"sources: {', '.join(session.sources)}   provider: {provider}")
    if provider != "rule":
        for role, model in config.resolved_models(provider).items():
            say(f"  {role:20s} {model}")
    say(
        "Tracks are Creative Commons. Every track that plays is credited in "
        "cache/credits.jsonl and out/SETLIST.md."
    )

    # The opening track is chosen without an agent and loaded synchronously, so
    # the set starts with audio rather than silence -- and so the ~14s numba JIT
    # warm-up for librosa is paid once, here.
    lo, hi = playable_window(config.DEFAULT_TARGET_BPM)
    say(f"looking for an opening track between {lo:.0f} and {hi:.0f} BPM...")
    candidates = session.find_candidates(lo, hi)
    if not candidates:
        say("Could not find an opening track.")
        if args.offline:
            say(f"With --offline you need audio in {args.crate}. Try --make-test-crate.")
        instrumentation.flush_tracing()
        return 1
    if not _prime_any(session, candidates, instrumentation, set_id):
        instrumentation.flush_tracing()
        return 1

    the_brain = brain_mod.make_brain(provider, session, say)
    runner = brain_mod.BrainRunner(the_brain, session, say)

    try:
        if args.render:
            _render(args, mixer, session, runner)
        else:
            _repl(args, mixer, session, runner)
    except KeyboardInterrupt:
        pass
    finally:
        from audio.mixer import Stop

        mixer.post(Stop())
        runner.stop()
        session.stop()
        if hasattr(the_brain, "close"):
            the_brain.close()
        say("")
        say(session.report())
        entries = library.read_credits()
        if entries:
            path = library.write_setlist()
            say(f"setlist: {path}  ({len(entries)} tracks)")
            unverified = sum(1 for e in entries if not e.license_verified)
            if unverified:
                say(f"{unverified} track(s) have an unverified license — see the setlist.")
        instrumentation.flush_tracing()
    return 0


def _repl(args: argparse.Namespace, mixer, session, runner) -> None:
    """The live path: open the stream, then take commands until quit."""
    import sounddevice as sd

    import music.library as library
    from audio.mixer import NudgeDeck

    stream = sd.OutputStream(
        samplerate=config.SR,
        blocksize=args.blocksize,
        channels=config.CHANNELS,
        dtype="float32",
        latency=args.latency,
        callback=mixer.sd_callback,
    )
    with stream:
        session.start()
        runner.start()
        say("")
        say(SLASH_HELP)
        say("")
        while True:
            line = _read_line()
            if line is None or line.strip() in {"/quit", "/q", "quit", "exit"}:
                return
            text = line.strip()
            if not text:
                continue

            if text in {"/help", "/h", "?"}:
                say(SLASH_HELP)
            elif text in {"/status", "/s"}:
                say(session.tool_get_now_playing())
            elif text == "/credits":
                entries = library.read_credits()
                if not entries:
                    say("nothing played yet")
                for i, e in enumerate(entries, start=1):
                    say(
                        f"  {i}. {e.title} — {e.artist} · {e.license_name}"
                        f"{library.license_flag(e.license_verified)}"
                    )
            elif text.startswith("/nudge"):
                parts = text.split()
                try:
                    ms = float(parts[1])
                except (IndexError, ValueError):
                    say("usage: /nudge -20   (milliseconds, negative pulls deck B earlier)")
                    continue
                frames = int(ms / 1000.0 * config.SR)
                mixer.post(NudgeDeck("b", frames))
                say(f"nudged deck B by {ms:+.0f} ms ({frames:+d} samples)")
            else:
                try:
                    session.q_user.put_nowait(text)
                except queue.Full:
                    say("hold on, still thinking about the last one")


def _render(args: argparse.Namespace, mixer, session, runner) -> None:
    """The offline path: drive the mixer from this loop and write a wav.

    Useful for two things the live REPL cannot give you: inspecting a transition
    in a waveform editor, and running the whole pipeline on a machine with no
    working audio output. The conductor is driven from here rather than from a
    timer, so the scheduling clock stays locked to the audio clock even though
    we render much faster than realtime.
    """
    import numpy as np
    import soundfile as sf

    from session import CONDUCTOR_TICK_S

    session.start(conductor=False)
    runner.start()

    def pending() -> bool:
        return (
            session.waiting_for_load()
            or not session.q_user.empty()
            or runner.busy.is_set()
        )

    total = int(args.duration * config.SR)
    tick_every = int(config.SR * CONDUCTOR_TICK_S)
    chunks: list[np.ndarray] = []
    done = 0
    next_tick = 0
    say(f"rendering {args.duration:.0f}s ...")

    while done < total:
        if done >= next_tick:
            session.tick()
            next_tick += tick_every

            # Rendering outruns both background threads, so pause the audio
            # clock while a download, a librosa pass, or a brain turn is still
            # in flight. Without this the renderer manufactures underruns and
            # empty queues that a live set never has.
            if pending() or mixer.snapshot.a_remaining_out_s < config.AUTOPILOT_LEAD_S:
                deadline = time.monotonic() + 60.0
                while time.monotonic() < deadline:
                    session.tick()
                    if not pending():
                        if mixer.snapshot.a_remaining_out_s >= config.AUTOPILOT_LEAD_S:
                            break
                        if session.queue_depth() > 0 and not session.waiting_for_load():
                            break
                        if session.faded_out():
                            break
                    time.sleep(0.05)

        n = min(args.blocksize, total - done)
        chunks.append(mixer.process(n).copy())
        done += n

    audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, config.CHANNELS))
    out_path = Path(args.render)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, config.SR)

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    say(f"wrote {out_path}  {len(audio) / config.SR:.1f}s  peak {peak:.3f}")
    if mixer.flip_log:
        say("transitions at:")
        for frame, track_id in mixer.flip_log:
            say(f"  {frame / config.SR:7.2f}s  -> {track_id}")


def run_seed_cache(args: argparse.Namespace) -> int:
    """Warm the cache so later sets can run --offline.

    Worth doing for ccMixter in particular: their TLS handshake takes 20-30
    seconds, so a set that has to search and download mid-run spends most of its
    time waiting. Seed once, then play from disk.
    """
    import instrumentation
    import music.library as library

    if args.offline:
        print("error: --seed-cache needs the network; drop --offline")
        return 2

    built = _build_session(args, "seed")
    if built is None:
        return 2
    session, _, _, _ = built

    want = args.seed_cache
    lo, hi = session.playable_window()
    say(f"seeding {want} track(s) between {lo:.0f} and {hi:.0f} BPM from: {', '.join(session.sources)}")

    done = 0
    for track in session.find_candidates(lo, hi, limit=want * 2):
        if done >= want:
            break
        try:
            record, _, _ = library.fetch_and_analyze(track, session.sources[track.source])
        except Exception as exc:
            say(f"  skipped {track.label}: {type(exc).__name__}: {exc}")
            continue
        done += 1
        say(
            f"  {done}/{want}  {track.label} — {record.bpm_used:.1f} BPM, "
            f"grid confidence {record.grid_confidence:.2f}"
        )

    instrumentation.flush_tracing()
    if done == 0:
        say("nothing could be seeded.")
        return 1
    say(f"cached {done} track(s). You can now run: dj.py --offline --source local,ccmixter")
    return 0


# ---------------------------------------------------------------- CLI


def _blocksize(text: str) -> int:
    """PortAudio accepts 0 for "host picks", but the mixer sizes its buffer from it."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("--blocksize must be a positive number of frames")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dj",
        description="A live, traced, multi-agent DJ that mixes royalty-free music.",
    )

    modes = p.add_argument_group("modes")
    modes.add_argument(
        "--selftest-mixer",
        action="store_true",
        help="offline DSP verification: crossfade correctness, no audio device needed",
    )
    modes.add_argument(
        "--trace-selftest",
        action="store_true",
        help=(
            "assert OTel context survives a thread hand-off, with a negative "
            "control (in-memory, no export)"
        ),
    )
    modes.add_argument(
        "--render",
        metavar="WAV",
        help="render a set to a wav file instead of playing it (no audio device needed)",
    )
    modes.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="seconds to render with --render (default: 180)",
    )

    audio = p.add_argument_group("audio")
    audio.add_argument(
        "--blocksize",
        type=_blocksize,
        default=config.BLOCKSIZE,
        help=f"frames per callback (default: {config.BLOCKSIZE})",
    )
    audio.add_argument(
        "--latency",
        default="high",
        choices=["low", "high"],
        help="PortAudio latency hint (default: high)",
    )

    agents = p.add_argument_group("agents")
    agents.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "claude", "openai", "rule"],
        help="which brain drives the set (default: auto)",
    )
    agents.add_argument("--model", help="override the model for every role")
    agents.add_argument("--model-host", dest="model_host")
    agents.add_argument("--model-crate-digger", dest="model_crate_digger")
    agents.add_argument("--model-transition-planner", dest="model_transition_planner")

    music = p.add_argument_group("music")
    music.add_argument(
        "--source",
        help="comma-separated: ccmixter,local (default: ccmixter, or local with --offline)",
    )
    music.add_argument(
        "--crate", type=Path, default=config.DEFAULT_CRATE_DIR, help="local library directory"
    )
    music.add_argument(
        "--offline", action="store_true", help="never touch the network; local sources only"
    )
    music.add_argument(
        "--make-test-crate",
        action="store_true",
        help="write synthetic CC0 tracks into the crate directory and exit",
    )
    music.add_argument(
        "--seed-cache",
        type=int,
        metavar="N",
        help=(
            "fetch and analyse N tracks from the networked sources, then exit. "
            "Run this once and every later set can use --offline."
        ),
    )

    p.add_argument_group("tracing").add_argument(
        "--trace-console", action="store_true", help="also print spans to the console"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config.set_model_overrides(
        {
            "host": args.model_host,
            "crate_digger": args.model_crate_digger,
            "transition_planner": args.model_transition_planner,
        },
        args.model,
    )

    if args.trace_selftest:
        import devtools.selftest as selftest

        return selftest.run_tracing()

    if args.selftest_mixer:
        import devtools.selftest as selftest

        return selftest.run()

    if args.make_test_crate:
        import devtools.testcrate as testcrate

        print(f"writing a synthetic CC0 crate into {args.crate}")
        return testcrate.write(args.crate)

    if args.seed_cache:
        return run_seed_cache(args)

    return run_set(args)


if __name__ == "__main__":
    raise SystemExit(main())
