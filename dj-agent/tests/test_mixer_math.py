"""The pure functions the crossfade depends on."""

from __future__ import annotations

import numpy as np
import pytest

import config
from audio.mixer import (
    Mixer,
    bars_to_samples,
    equal_power_curves,
    linear_fade_out,
    next_bar_after,
    rate_ramp,
)


def test_bars_to_samples_matches_hand_arithmetic():
    # 8 bars of 4 beats at 122 BPM = 8*4*(60/122) seconds
    assert bars_to_samples(8, 122.0) == round(8 * 4 * (60 / 122) * config.SR)


def test_bars_to_samples_rejects_nonsense_tempo():
    with pytest.raises(ValueError):
        bars_to_samples(8, 0.0)


def test_equal_power_curves_preserve_power():
    gain_a, gain_b = equal_power_curves(4096)
    total = gain_a.astype(np.float64) ** 2 + gain_b.astype(np.float64) ** 2
    assert np.allclose(total, 1.0, atol=1e-6)
    assert gain_a[0] == pytest.approx(1.0)
    assert gain_b[-1] == pytest.approx(1.0)
    assert gain_a.dtype == np.float32


def test_fade_and_ramp_endpoints():
    fade = linear_fade_out(100)
    assert fade[0] == pytest.approx(1.0)
    assert fade[-1] == pytest.approx(0.0)
    ramp = rate_ramp(1.0, 0.9375, 100)
    assert ramp[0] == pytest.approx(1.0)
    assert ramp[-1] == pytest.approx(0.9375)


def test_next_bar_after_is_strictly_greater():
    grid = np.array([0, 100, 200, 300], dtype=np.int64)
    assert next_bar_after(grid, 0) == 100
    assert next_bar_after(grid, 99.9) == 100
    assert next_bar_after(grid, 100) == 200  # strictly greater, not >=
    assert next_bar_after(grid, 300) is None
    assert next_bar_after(np.zeros(0, dtype=np.int64), 5) is None


def test_process_returns_a_scratch_view_that_callers_must_copy():
    """Documents the aliasing contract, so nobody rediscovers it the hard way."""
    mixer = Mixer()
    from audio.mixer import LoadDeckB

    ramp = np.linspace(0.0, 1.0, config.SR, dtype=np.float32)
    mixer.load_deck_a(
        LoadDeckB(
            track_id="ramp",
            buf=np.stack([ramp, ramp], axis=1),
            grid=np.zeros(1, dtype=np.int64),
            rate=1.0,
            start_pos=0.0,
        )
    )
    first = mixer.process(512)
    second = mixer.process(512)
    assert first.base is second.base  # same scratch allocation
    assert np.shares_memory(first, second)


def test_varispeed_read_changes_duration_not_content():
    """A deck played at rate r consumes r samples of source per output sample."""
    mixer = Mixer()
    n = config.SR
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    buf = np.stack([ramp, ramp], axis=1)
    from audio.mixer import LoadDeckB

    mixer.load_deck_a(
        LoadDeckB(
            track_id="ramp", buf=buf, grid=np.zeros(1, dtype=np.int64), rate=0.5, start_pos=0.0
        )
    )
    out = np.concatenate([mixer.process(1024).copy() for _ in range(8)], axis=0)
    # At rate 0.5 the deck has advanced half as far through the source.
    assert mixer.a.pos == pytest.approx(0.5 * 8 * 1024)
    # And the output is a stretched version of the same monotonic ramp.
    assert np.all(np.diff(out[:, 0]) >= -1e-6)


def test_callback_never_clips_when_both_decks_are_hot():
    mixer = Mixer()
    from audio.mixer import LoadDeckB, StartXfade

    loud = np.ones((config.SR, config.CHANNELS), dtype=np.float32)
    mixer.load_deck_a(
        LoadDeckB(track_id="a", buf=loud, grid=np.zeros(1, dtype=np.int64), rate=1.0, start_pos=0.0)
    )
    mixer.post(
        LoadDeckB(track_id="b", buf=loud, grid=np.zeros(1, dtype=np.int64), rate=1.0, start_pos=0.0)
    )
    gain_a, gain_b = equal_power_curves(8192)
    mixer.post(StartXfade(delay_frames=0, gain_a=gain_a, gain_b=gain_b))
    out = np.concatenate([mixer.process(1024).copy() for _ in range(8)], axis=0)
    assert float(np.abs(out).max()) <= 1.0


def _stereo_ones(seconds: float = 4.0) -> np.ndarray:
    return np.ones((int(config.SR * seconds), config.CHANNELS), dtype=np.float32)


def test_a_second_crossfade_mid_fade_is_refused_not_crashed():
    """Regression: a StartXfade arriving mid-fade used to crash the audio thread.

    With the second command's delay suppressing the crossfade's own segment
    bound, the render loop asked for a full block while fewer gain samples were
    left, and raised `operands could not be broadcast together with shapes
    (1024,2) (428,1)`. That aborted the PortAudio stream and left the set silent.
    Two independent conductor paths can post the second command.
    """
    from audio.mixer import LoadDeckB, StartXfade

    mixer = Mixer()
    mixer.load_deck_a(
        LoadDeckB(
            track_id="a",
            buf=_stereo_ones(),
            grid=np.zeros(1, dtype=np.int64),
            rate=1.0,
            start_pos=0.0,
        )
    )
    mixer.post(
        LoadDeckB(
            track_id="b",
            buf=_stereo_ones(),
            grid=np.zeros(1, dtype=np.int64),
            rate=1.0,
            start_pos=0.0,
        )
    )
    # A fade whose tail is shorter than one block, which is what the crash needed.
    gain_a, gain_b = equal_power_curves(3500)
    mixer.post(StartXfade(delay_frames=0, gain_a=gain_a, gain_b=gain_b))
    for _ in range(3):
        mixer.process(1024)
    assert mixer._xfade_len - mixer._xfade_pos == 428

    mixer.post(StartXfade(delay_frames=2048, gain_a=gain_a, gain_b=gain_b))
    flips_before = mixer.snapshot.flips
    out = mixer.process(1024)
    assert np.isfinite(out).all()
    assert mixer._dropped_xfades == 1  # refused, so the illegal state never forms
    assert mixer.snapshot.flips == flips_before  # and no early flip

    # The segment bound is the second defence: it has to hold even if a pending
    # delay and a running crossfade ever coexist again.
    mixer._xfade_pos, mixer._xfade_len = 0, 3500
    mixer._xfade_gain_a, mixer._xfade_gain_b = gain_a, gain_b
    mixer._pending_delay = 4096
    # Four blocks, so the last one lands on the fade's short tail.
    for _ in range(4):
        assert np.isfinite(mixer.process(1024)).all()


def test_the_callback_survives_a_failure_and_reports_it():
    """An exception on the audio thread aborts the stream, so it must not escape."""
    mixer = Mixer()
    outdata = np.ones((1024, config.CHANNELS), dtype=np.float32)

    def boom(frames: int):
        raise ValueError("synthetic render failure")

    mixer.process = boom  # type: ignore[method-assign]
    mixer.sd_callback(outdata, 1024, None, None)

    assert not outdata.any()  # silence, not the previous block's audio
    assert mixer.snapshot.callback_errors == 1
    assert "synthetic render failure" in (mixer.snapshot.callback_error or "")
