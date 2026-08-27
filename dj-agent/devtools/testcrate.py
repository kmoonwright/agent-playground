"""Generate a synthetic local crate so the whole pipeline runs with no network.

The tracks are simple four-on-the-floor loops (kick, hat, bass, pad) at exact
tempos, which gives librosa real onsets to track instead of the artificial
impulses the DSP selftest uses. They are written as FLAC because mutagen can
write a BPM tag into Vorbis comments, and one track is left deliberately
untagged to exercise the "source has no BPM, analysis must compute it" path.

We synthesize these ourselves, so the accompanying LICENSE.json marks them CC0 —
that is an honest claim, and it also exercises the verified-license branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import SR

SPEC = [
    # (filename, bpm, root_hz, write_bpm_tag)
    ("dusk_runner_124.flac", 124.0, 55.0, True),
    ("neon_transit_120.flac", 120.0, 49.0, True),
    ("slow_tide_100.flac", 100.0, 41.2, True),
    ("hazy_dub_122.flac", 122.0, 61.7, True),
    ("untagged_mystery.flac", 126.0, 58.3, False),
]


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    freq = 110.0 * np.exp(-t * 28.0) + 45.0
    env = np.exp(-t * 9.0)
    return (np.sin(2 * np.pi * np.cumsum(freq) / SR) * env).astype(np.float32)


def _hat(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    env = np.exp(-t * 60.0)
    return (rng.standard_normal(n).astype(np.float32) * env * 0.35).astype(np.float32)


def build_track(bpm: float, root_hz: float, seconds: float = 75.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    beat = SR * 60.0 / bpm
    out = np.zeros(n, dtype=np.float32)

    kick = _kick(int(0.35 * SR))
    hat = _hat(int(0.08 * SR), rng)

    beats = int(n / beat) - 1
    for b in range(beats):
        pos = int(round(b * beat))
        out[pos : pos + len(kick)] += kick * 0.9
        off = int(round((b + 0.5) * beat))
        out[off : off + len(hat)] += hat

    # Bass walks a 4-bar root movement; the pad doubles it two octaves up,
    # which is enough harmonic content for a beat tracker to lock onto.
    t = np.arange(n) / SR
    degrees = np.array([1.0, 1.0, 6 / 5, 4 / 3])
    bar = beat * 4
    bass = np.zeros(n, dtype=np.float32)
    pad = np.zeros(n, dtype=np.float32)
    for i in range(int(n / bar) + 1):
        s, e = int(i * bar), min(n, int((i + 1) * bar))
        if s >= e:
            break
        f = root_hz * degrees[i % len(degrees)]
        seg = t[s:e]
        bass[s:e] = 0.35 * np.sin(2 * np.pi * f * seg).astype(np.float32)
        pad[s:e] = (
            0.10 * np.sin(2 * np.pi * f * 3 * seg) + 0.07 * np.sin(2 * np.pi * f * 4 * seg)
        ).astype(np.float32)

    # Gentle bass envelope on each beat so the low end doesn't drone.
    pulse = np.zeros(n, dtype=np.float32)
    for b in range(beats):
        pos = int(round(b * beat))
        seg = min(int(beat), n - pos)
        pulse[pos : pos + seg] = np.exp(-np.arange(seg) / (0.25 * SR)).astype(np.float32)
    out += bass * (0.4 + 0.6 * pulse) + pad

    peak = float(np.abs(out).max()) or 1.0
    out = (out / peak * 0.7).astype(np.float32)
    return np.stack([out, out], axis=1)


def write_track(
    path: Path,
    bpm: float,
    root_hz: float,
    *,
    tag_bpm: bool,
    seconds: float,
    seed: int = 7,
    title: str | None = None,
    artist: str = "dj-agent testcrate",
) -> Path:
    """Write one tagged FLAC. Shared with the test fixtures."""
    import soundfile as sf
    from mutagen.flac import FLAC

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, build_track(bpm, root_hz, seconds=seconds, seed=seed), SR, format="FLAC")

    meta = FLAC(path)
    meta["title"] = title or path.stem.replace("_", " ").title()
    meta["artist"] = artist
    if tag_bpm:
        meta["bpm"] = f"{bpm:g}"
    meta.save()
    return path


def write(crate_dir: Path, seconds: float = 75.0) -> int:
    crate_dir.mkdir(parents=True, exist_ok=True)
    licenses: dict[str, dict[str, str]] = {}

    for i, (name, bpm, root, tag_bpm) in enumerate(SPEC):
        write_track(
            crate_dir / name, bpm, root, tag_bpm=tag_bpm, seconds=seconds, seed=7 + i
        )
        licenses[name] = {
            "license_name": "CC0 1.0 (synthesized by dj-agent)",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "page_url": "https://github.com/  (local test crate)",
        }
        tagged = f"bpm tag {bpm:g}" if tag_bpm else "NO bpm tag (analysis must compute it)"
        print(f"  {name:26s} {bpm:5.1f} BPM  {seconds:.0f}s  {tagged}")

    (crate_dir / "LICENSE.json").write_text(json.dumps(licenses, indent=2))
    print(f"  LICENSE.json               {len(licenses)} entries (marks these verified)")
    return 0
