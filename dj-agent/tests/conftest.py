"""Shared fixtures. Everything here is offline: no network, no audio device, no API key."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the whole library at a temp directory so tests never touch cache/."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "TRACKS_DIR", tmp_path / "cache" / "tracks")
    monkeypatch.setattr(config, "API_CACHE_DIR", tmp_path / "cache" / "api")
    monkeypatch.setattr(config, "CREDITS_PATH", tmp_path / "cache" / "credits.jsonl")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    config.ensure_dirs()


@pytest.fixture(autouse=True)
def clean_model_overrides(monkeypatch):
    for name in ("DJ_MODEL", "DJ_MODEL_HOST", "DJ_MODEL_CRATE_DIGGER", "DJ_MODEL_TRANSITION_PLANNER"):
        monkeypatch.delenv(name, raising=False)
    config.set_model_overrides({r: None for r in config.ROLES}, None)
    yield
    config.set_model_overrides({r: None for r in config.ROLES}, None)


@pytest.fixture
def crate(tmp_path):
    """A two-track crate: one BPM-tagged, one deliberately untagged.

    Written with `testcrate.write_track`, the same helper `--make-test-crate`
    uses, so there is one implementation of "synthesize a tagged FLAC".
    """
    import devtools.testcrate as testcrate

    directory = tmp_path / "crate"
    for name, bpm, tag in (("tagged_120", 120.0, True), ("untagged", 124.0, False)):
        testcrate.write_track(
            directory / f"{name}.flac",
            bpm,
            55.0,
            tag_bpm=tag,
            seconds=12.0,
            title=name,
            artist="fixture",
        )
    return directory


@pytest.fixture
def local_source(crate):
    from music.local import LocalSource

    return LocalSource(crate)


@pytest.fixture
def fixture_wav(tmp_path):
    """A short, exactly-120-BPM click file for analysis determinism tests."""
    import soundfile as sf

    from devtools.selftest import click_track

    audio, _ = click_track(120.0, 10.0)
    path = tmp_path / "clicks_120.wav"
    sf.write(path, audio, config.SR)
    return path


@pytest.fixture
def analysed_track(local_source):
    """The untagged crate track, fetched into the cache and analysed.

    Five guardrail tests need "a second track that is ready to queue"; this is
    that setup, once.
    """
    import audio.analysis as analysis
    import music.library as library

    track = sorted(local_source.scan(), key=lambda t: t.title)[-1]
    path = local_source.fetch(track, library.track_dir(track.track_id))
    library.save_track(track)
    record, _ = analysis.ensure_analyzed(track, path)
    return track, record


@pytest.fixture
def session(local_source, monkeypatch):
    """A DJSession with a real mixer but no threads started and no audio device."""
    from audio.mixer import Mixer
    from session import DJSession

    return DJSession(
        mixer=Mixer(),
        sources={"local": local_source},
        say=lambda _: None,
        set_id="set-test",
    )


@pytest.fixture
def primed_session(session, local_source):
    """A session with deck A actually loaded, so tools have real state to report."""
    tracks = sorted(local_source.scan(), key=lambda t: t.title)
    session.prime(tracks[0])
    return session

