"""Decode, resample, loudness-normalize, and beat-track one file. Pure and offline.

The BPM used for playback rate is the *source's* metadata BPM whenever there is
one, not librosa's estimate. A 0.3% tempo error accumulates ~48 ms of drift over
a 16-second crossfade, which flams audibly; producer-authored BPM is usually
exact, where a beat tracker's float is not. librosa supplies the *phase* -- where
the beats sit -- and supplies the tempo only when the source has none.

There is no reliable downbeat tracker, so bars are assumed 4/4 and taken every
fourth detected beat. `grid_confidence` records how much the two BPM sources
agree, so the crate digger can prefer tracks whose grid we trust.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import config
import music.library as library
from schema import ANALYSIS_SCHEMA_VERSION, Analysis
from music.source import Track

# Octave and metric errors a beat tracker commonly makes; treated as agreement.
BPM_FACTORS = (1.0, 2.0, 0.5, 1.5, 2.0 / 3.0)
CONFIDENCE_TOLERANCE = 0.05  # 5% off -> confidence 0


def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def decode(path: Path) -> tuple[np.ndarray, int]:
    """Read any libsndfile-supported file as float32 (n, channels).

    soundfile 0.14 bundles a libsndfile that decodes mp3 natively, which is why
    this project needs no ffmpeg.

    The duration cap is enforced here rather than upstream because the byte cap
    alone does not bound it -- 20 MB of low-bitrate mp3 is over an hour of audio,
    and `to_stereo`/`resample`/`normalize` each copy the decoded buffer.
    """
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate and info.frames / info.samplerate > config.MAX_TRACK_SECONDS:
        raise ValueError(
            f"{path.name} declares {info.frames / info.samplerate:.0f}s, "
            f"over the {config.MAX_TRACK_SECONDS}s cap"
        )
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # The header is untrusted, so re-check what actually decoded.
    if sr and audio.shape[0] / sr > config.MAX_TRACK_SECONDS:
        raise ValueError(
            f"{path.name} decoded to {audio.shape[0] / sr:.0f}s, "
            f"over the {config.MAX_TRACK_SECONDS}s cap"
        )
    return audio, int(sr)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    if audio.shape[1] > 2:
        return np.ascontiguousarray(audio[:, :2])
    return audio


def resample(audio: np.ndarray, sr_in: int, sr_out: int = config.SR) -> np.ndarray:
    """Resample on the loader thread so deck rate only ever carries tempo."""
    if sr_in == sr_out:
        return audio
    import librosa

    out = librosa.resample(audio.T, orig_sr=sr_in, target_sr=sr_out, axis=-1)
    return np.ascontiguousarray(out.T.astype(np.float32))


def normalize(audio: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Scale toward TARGET_RMS_DBFS with a peak ceiling. Returns (audio, rms_dbfs, peak, gain)."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) or 1e-9
    rms_dbfs = 20.0 * float(np.log10(rms))
    gain = 10.0 ** ((config.TARGET_RMS_DBFS - rms_dbfs) / 20.0)
    peak = float(np.abs(audio).max()) or 1e-9
    if peak * gain > config.PEAK_CEILING:
        gain = config.PEAK_CEILING / peak
    out = (audio * gain).astype(np.float32)
    return out, rms_dbfs, peak, float(gain)


def bpm_agreement(bpm_meta: float | None, bpm_librosa: float | None) -> float:
    """1.0 when the two BPM sources agree (allowing octave errors), 0.0 when not."""
    if not bpm_meta or not bpm_librosa:
        return 0.0
    best = min(abs(bpm_librosa * f - bpm_meta) / bpm_meta for f in BPM_FACTORS)
    return float(max(0.0, 1.0 - best / CONFIDENCE_TOLERANCE))


def grid_regularity(beat_samples: np.ndarray) -> float:
    """Fallback confidence when there is no metadata BPM: how even the beats are."""
    if beat_samples.size < 4:
        return 0.0
    intervals = np.diff(beat_samples.astype(np.float64))
    mean = float(intervals.mean())
    if mean <= 0:
        return 0.0
    cv = float(intervals.std() / mean)
    return float(max(0.0, 1.0 - cv / CONFIDENCE_TOLERANCE))


def beat_track(audio: np.ndarray, sr: int = config.SR) -> tuple[float, np.ndarray]:
    import librosa

    mono = audio.mean(axis=1).astype(np.float32)
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, units="samples")
    tempo_val = float(np.atleast_1d(tempo)[0])
    return tempo_val, np.asarray(beats, dtype=np.int64)


def analyze(
    track: Track, source_path: Path, file_sha1: str | None = None
) -> tuple[Analysis, np.ndarray]:
    """Full analysis. Returns the record and the deck-ready stereo buffer."""
    import librosa

    file_sha1 = file_sha1 or sha1_of_file(source_path)

    raw, sr_in = decode(source_path)
    audio = resample(to_stereo(raw), sr_in)
    audio, rms_dbfs, _peak, _gain = normalize(audio)

    bpm_librosa, beats = beat_track(audio)
    bpm_meta = float(track.bpm) if track.bpm else None

    if bpm_meta:
        bpm_used, bpm_source = bpm_meta, "metadata"
        confidence = bpm_agreement(bpm_meta, bpm_librosa)
    else:
        bpm_used, bpm_source = bpm_librosa, "librosa"
        confidence = grid_regularity(beats)

    # 4/4 assumed: no dependable downbeat tracker exists, so "bar" here means
    # "every fourth detected beat" and the chosen downbeat is arbitrary. The
    # /nudge command exists because of exactly this.
    bars = beats[:: config.BEATS_PER_BAR]

    analysis = Analysis(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        track_id=track.track_id,
        sr=config.SR,
        n_samples=int(audio.shape[0]),
        duration_s=float(audio.shape[0] / config.SR),
        bpm_meta=bpm_meta,
        bpm_librosa=float(bpm_librosa),
        bpm_used=float(bpm_used),
        bpm_source=bpm_source,  # type: ignore[arg-type]
        beat_samples=[int(b) for b in beats],
        bar_indices=[int(b) for b in bars],
        grid_confidence=round(float(confidence), 4),
        rms_dbfs=round(rms_dbfs, 2),
        analyzed_at=library.now_iso(),
        librosa_version=librosa.__version__,
        file_sha1=file_sha1,
    )
    return analysis, audio


def save(analysis: Analysis, audio: np.ndarray) -> None:
    library.save_analysis(analysis)
    np.save(library.audio_path(analysis.track_id), audio, allow_pickle=False)


def load_audio(track_id: str) -> np.ndarray | None:
    path = library.audio_path(track_id)
    if not path.exists():
        return None
    try:
        audio = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if audio.ndim != 2 or audio.shape[1] != config.CHANNELS:
        return None
    return np.ascontiguousarray(audio.astype(np.float32))


GRID_TRUST_THRESHOLD = 0.5


def bar_grid(analysis: Analysis) -> np.ndarray:
    """Bar-boundary sample indices, with a fallback for untrustworthy beat tracking.

    On real music librosa's tempo estimate is often an octave or metric level
    away from the producer's BPM -- a 120 BPM house track can beat-track at
    161.5. When that happens the *spacing* of the detected beats is wrong, so
    taking every fourth one produces a grid that drifts. But the metadata BPM is
    still reliable, and the first detected beat is still a plausible onset, so we
    rebuild the grid arithmetically from the trusted tempo anchored to that
    onset. That turns a wrong grid into a merely-imperfectly-phased one, which
    the /nudge command can fix by ear.
    """
    if _grid_is_trusted(analysis):
        return np.asarray(analysis.bar_indices, dtype=np.int64)
    return synthetic_bar_grid(analysis)


def _grid_is_trusted(analysis: Analysis) -> bool:
    """One condition, so `grid_source` cannot describe a grid we did not build."""
    return analysis.grid_confidence >= GRID_TRUST_THRESHOLD or not analysis.bpm_meta


def synthetic_bar_grid(analysis: Analysis) -> np.ndarray:
    """Even bar grid at the metadata BPM, anchored to the first detected beat."""
    bpm = analysis.bpm_meta or analysis.bpm_used
    bar_len = analysis.sr * 60.0 / bpm * config.BEATS_PER_BAR
    anchor = float(analysis.beat_samples[0]) if analysis.beat_samples else 0.0
    count = int(max(0.0, (analysis.n_samples - anchor)) // bar_len)
    if count <= 0:
        return np.asarray([int(anchor)], dtype=np.int64)
    return (anchor + np.arange(count, dtype=np.float64) * bar_len).astype(np.int64)


def grid_source(analysis: Analysis) -> str:
    return "beat_track" if _grid_is_trusted(analysis) else "synthetic_from_metadata_bpm"


def ensure_analyzed(track: Track, source_path: Path) -> tuple[Analysis, np.ndarray]:
    """Reuse the cached analysis when the file hasn't changed, else recompute."""
    file_sha1 = sha1_of_file(source_path)
    cached = library.load_analysis(track.track_id, file_sha1=file_sha1)
    audio = load_audio(track.track_id) if cached else None
    if cached is not None and audio is not None:
        return cached, audio

    # The digest is passed through so the file is read once, not twice.
    analysis, audio = analyze(track, source_path, file_sha1)
    save(analysis, audio)
    return analysis, audio
