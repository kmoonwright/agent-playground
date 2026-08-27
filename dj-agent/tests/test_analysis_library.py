"""Analysis determinism, the BPM decision, the grid fallback, and credits."""

from __future__ import annotations

import numpy as np
import pytest

import audio.analysis as analysis
import config
import music.library as library
from schema import ANALYSIS_SCHEMA_VERSION
from music.source import Track


def _track(path, bpm=None) -> Track:
    from music.source import track_id_for

    return Track(
        source="local",
        track_id=track_id_for(str(path)),
        title=path.stem,
        artist="fixture",
        fetch_uri=str(path),
        bpm=bpm,
        bpm_source="metadata" if bpm else "unknown",
    )


def test_analysis_is_deterministic(fixture_wav):
    track = _track(fixture_wav, bpm=120.0)
    first, audio_a = analysis.analyze(track, fixture_wav)
    second, audio_b = analysis.analyze(track, fixture_wav)
    assert first.bpm_librosa == second.bpm_librosa
    assert first.beat_samples == second.beat_samples
    assert first.file_sha1 == second.file_sha1
    assert np.array_equal(audio_a, audio_b)


def test_metadata_bpm_wins_over_the_beat_tracker(fixture_wav):
    """Rate comes from producer BPM; a 0.3% error flams over a 16s crossfade."""
    track = _track(fixture_wav, bpm=120.0)
    record, _ = analysis.analyze(track, fixture_wav)
    assert record.bpm_used == 120.0
    assert record.bpm_source == "metadata"


def test_librosa_supplies_bpm_when_the_source_has_none(fixture_wav):
    track = _track(fixture_wav, bpm=None)
    record, _ = analysis.analyze(track, fixture_wav)
    assert record.bpm_source == "librosa"
    assert record.bpm_used == pytest.approx(120.0, abs=3.0)


def test_bpm_agreement_forgives_octave_errors():
    assert analysis.bpm_agreement(120.0, 120.0) == pytest.approx(1.0)
    assert analysis.bpm_agreement(120.0, 240.0) == pytest.approx(1.0)  # double time
    assert analysis.bpm_agreement(120.0, 60.0) == pytest.approx(1.0)  # half time
    assert analysis.bpm_agreement(120.0, 161.5) == 0.0  # genuinely disagrees
    assert analysis.bpm_agreement(None, 120.0) == 0.0


def test_grid_falls_back_to_arithmetic_when_beat_tracking_is_untrusted(fixture_wav):
    """The real ccMixter case: 120 BPM metadata, 161.5 BPM beat track."""
    track = _track(fixture_wav, bpm=120.0)
    record, _ = analysis.analyze(track, fixture_wav)
    broken = record.model_copy(update={"grid_confidence": 0.0, "bpm_meta": 120.0})

    assert analysis.grid_source(broken) == "synthetic_from_metadata_bpm"
    grid = analysis.bar_grid(broken)
    spacing = np.diff(grid)
    expected = config.SR * 60.0 / 120.0 * config.BEATS_PER_BAR
    assert np.allclose(spacing, expected, atol=1)  # perfectly even, unlike the detected grid

    trusted = record.model_copy(update={"grid_confidence": 0.9})
    assert analysis.grid_source(trusted) == "beat_track"
    assert list(analysis.bar_grid(trusted)) == trusted.bar_indices


def test_normalize_targets_rms_when_the_peak_allows_it():
    """A constant 0.9 signal is far louder than -14 dBFS, so gain reduces it."""
    loud = np.full((1000, 2), 0.9, dtype=np.float32)
    out, rms_dbfs, peak, gain = analysis.normalize(loud)
    assert rms_dbfs == pytest.approx(-0.915, abs=0.01)
    assert peak == pytest.approx(0.9)
    assert gain < 1.0
    achieved = 20 * np.log10(float(np.sqrt(np.mean(np.square(out.astype(np.float64))))))
    assert achieved == pytest.approx(config.TARGET_RMS_DBFS, abs=0.01)


def test_normalize_respects_the_peak_ceiling():
    """Low RMS with a big transient: the RMS target would clip, so the ceiling wins."""
    spiky = np.zeros((10_000, 2), dtype=np.float32)
    spiky[0] = 0.9  # one loud transient, near-silence elsewhere
    out, rms_dbfs, peak, gain = analysis.normalize(spiky)
    assert rms_dbfs < config.TARGET_RMS_DBFS  # quiet enough that gain wants to be > 1
    assert peak == pytest.approx(0.9)
    assert gain == pytest.approx(config.PEAK_CEILING / 0.9, rel=1e-3)
    assert float(np.abs(out).max()) <= config.PEAK_CEILING + 1e-6


def test_cached_analysis_is_rejected_when_the_file_changes(fixture_wav):
    track = _track(fixture_wav, bpm=120.0)
    record, audio = analysis.ensure_analyzed(track, fixture_wav)
    library.save_track(track)
    assert library.load_analysis(track.track_id, file_sha1=record.file_sha1) is not None
    assert library.load_analysis(track.track_id, file_sha1="0" * 40) is None


def test_cached_analysis_is_rejected_on_schema_bump(fixture_wav, monkeypatch):
    track = _track(fixture_wav, bpm=120.0)
    record, _ = analysis.ensure_analyzed(track, fixture_wav)
    stale = record.model_copy(update={"schema_version": ANALYSIS_SCHEMA_VERSION - 1})
    library.save_analysis(stale)
    assert library.load_analysis(track.track_id) is None


def test_is_ready_requires_meta_analysis_and_audio(fixture_wav):
    track = _track(fixture_wav, bpm=120.0)
    assert library.is_ready(track.track_id) is False
    library.save_track(track)
    analysis.ensure_analyzed(track, fixture_wav)
    assert library.is_ready(track.track_id) is True


def test_credits_record_unverified_licenses_and_the_setlist_flags_them(fixture_wav):
    track = _track(fixture_wav, bpm=120.0)
    track = Track(**{**track.__dict__, "license_name": "local (unverified)"})
    library.save_track(track)
    library.append_credit(library.credit_for(track, 120.0, 42.0))

    entries = library.read_credits()
    assert len(entries) == 1
    assert entries[0].license_verified is False
    assert entries[0].seconds_played == 42.0

    text = library.write_setlist().read_text()
    assert "license unverified" in text
    assert "1 of 1 tracks have an unverified license" in text


def test_search_cache_ranks_by_distance_from_the_midpoint(fixture_wav, tmp_path):
    import soundfile as sf

    from devtools.selftest import click_track

    for bpm in (118.0, 122.0, 130.0):
        audio, _ = click_track(bpm, 4.0)
        path = tmp_path / f"t{bpm:.0f}.wav"
        sf.write(path, audio, config.SR)
        track = _track(path, bpm=bpm)
        library.save_track(track)

    found = library.search_cache(115, 135, limit=3)
    # Midpoint is 125, so 122 is closest, then 130, then 118.
    assert [t.bpm for t in found] == [122.0, 130.0, 118.0]


def test_fetch_and_analyze_is_the_one_path_from_track_to_buffer(local_source, crate):
    """prime, the loader thread, and cache seeding all go through this helper,
    so its cache-hit behaviour is the cache-hit behaviour of all three."""
    track = local_source.scan()[0]

    record, audio, cached = library.fetch_and_analyze(track, local_source)
    assert cached is False, "first call has to fetch"
    assert audio.shape[1] == config.CHANNELS
    assert library.is_ready(track.track_id)

    again, audio_again, cached_again = library.fetch_and_analyze(track, local_source)
    assert cached_again is True, "second call must come off disk"
    assert again.file_sha1 == record.file_sha1
    assert np.array_equal(audio, audio_again)


def test_fetch_and_analyze_is_used_by_prime_and_the_loader(session, local_source, monkeypatch):
    """A regression guard: if a caller stops using the helper, the paths diverge."""
    calls: list[str] = []
    real = library.fetch_and_analyze

    def spy(track, source):
        calls.append(track.track_id)
        return real(track, source)

    monkeypatch.setattr(library, "fetch_and_analyze", spy)

    track = local_source.scan()[0]
    session.prime(track)
    assert calls == [track.track_id]
