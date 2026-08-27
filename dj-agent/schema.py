"""Pydantic v2 models: tool arguments, analysis records, and credit lines.

Every LLM-supplied argument is validated through a model here before it reaches
`DJSession`, so a hallucinated field or an out-of-range number becomes a tool
observation the agent can react to rather than a traceback. The transition plan
in particular is *clamped* here rather than trusted — the validator, not the
model, decides what the mixer is allowed to do. `validate_plan` at the bottom is
that guardrail.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import BEATS_PER_BAR, DEFAULT_CROSSFADE_BARS, RATE_MAX, RATE_MIN

BpmSource = Literal["metadata", "librosa", "unknown"]

# Every track id in the system is `music.source.track_id_for`'s output: the first
# 12 hex digits of a sha1. Constraining it here is what stops an LLM-supplied id
# from becoming a path — `library.track_dir` joins it onto the cache directory.
TrackId = Annotated[str, Field(pattern=r"^[0-9a-f]{12}$")]

# Bounds on a BPM the agents may ask for. Restated as JSON for the tool schemas
# in `agents/tools.py`, which imports these rather than repeating the numbers.
BPM_ARG_MIN, BPM_ARG_MAX = 40, 220

ANALYSIS_SCHEMA_VERSION = 3


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- library records


class TrackRef(Strict):
    """The compact shape a candidate takes when shown to an agent."""

    track_id: TrackId
    source: str
    title: str
    artist: str
    bpm: float | None = None
    bpm_source: BpmSource = "unknown"
    duration_s: float | None = None
    license_name: str | None = None
    license_verified: bool = False
    cached: bool = False
    bpm_delta_pct: float | None = None
    grid_confidence: float | None = None


class Analysis(Strict):
    """Everything the mixer needs about a track, computed once and cached."""

    schema_version: int = ANALYSIS_SCHEMA_VERSION
    track_id: TrackId
    sr: int
    n_samples: int
    duration_s: float
    bpm_meta: float | None
    bpm_librosa: float | None
    bpm_used: float
    bpm_source: BpmSource
    beat_samples: list[int]
    bar_indices: list[int]
    grid_confidence: float
    rms_dbfs: float
    analyzed_at: str
    librosa_version: str
    file_sha1: str


class CreditEntry(Strict):
    """One line of credits.jsonl, written when a deck actually starts sounding."""

    played_at: str
    track_id: TrackId
    source: str
    title: str
    artist: str
    license_name: str | None
    license_url: str | None
    page_url: str | None
    fetch_uri: str
    license_verified: bool
    bpm_used: float
    seconds_played: float | None = None


# ---------------------------------------------------------------- host tool args


class QueueTrackArgs(Strict):
    track_id: TrackId
    crossfade_bars: int = Field(default=DEFAULT_CROSSFADE_BARS, ge=1, le=32)
    cue_seconds: float = Field(default=0.0, ge=0.0)
    position: Literal["next", "end"] = "next"


class SetTargetBpmArgs(Strict):
    bpm: float = Field(ge=60.0, le=180.0)
    ramp_bars: int = Field(default=4, ge=1, le=32)


class SkipArgs(Strict):
    crossfade_bars: int = Field(default=2, ge=1, le=32)


class TempoRange(Strict):
    """An ordered BPM range. Shared by the two search-shaped tool arg models."""

    bpm_min: int = Field(ge=BPM_ARG_MIN, le=BPM_ARG_MAX)
    bpm_max: int = Field(ge=BPM_ARG_MIN, le=BPM_ARG_MAX)

    @field_validator("bpm_max")
    @classmethod
    def _ordered(cls, v: int, info) -> int:
        low = info.data.get("bpm_min")
        if low is not None and v < low:
            raise ValueError("bpm_max must be >= bpm_min")
        return v


class FindTracksArgs(TempoRange):
    brief: str = Field(min_length=1, max_length=400)
    limit: int = Field(default=3, ge=1, le=5)


class PlanTransitionArgs(Strict):
    to_track_id: TrackId
    style_hint: str | None = Field(default=None, max_length=200)


# ---------------------------------------------------------------- sub-agent args


class SearchSourceArgs(TempoRange):
    source: str
    tags: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=8, ge=1, le=12)


class InspectTrackArgs(Strict):
    track_id: TrackId


class Candidate(Strict):
    track_id: TrackId
    reason: str = Field(min_length=1, max_length=300)


class Candidates(Strict):
    """Terminal submit tool for crate_digger."""

    candidates: list[Candidate] = Field(min_length=1, max_length=5)


class GetAnalysisArgs(Strict):
    track_id: TrackId


class TransitionPlan(Strict):
    """Terminal submit tool for transition_planner. Advisory until validated."""

    crossfade_bars: int = Field(ge=1, le=32)
    cue_seconds: float = Field(ge=0.0)
    target_bpm: float = Field(ge=60.0, le=180.0)
    note: str = Field(default="", max_length=300)


class ValidatedPlan(Strict):
    """What the guardrail hands the mixer, plus what it had to change.

    `crossfade_bars`, `cue_seconds`, and `target_bpm` are what the mixer is
    actually driven with. `rate_a`/`rate_b` are reported for the trace and for
    tests -- only deck A is rate-ramped, from `target_bpm`.
    """

    crossfade_bars: int
    cue_seconds: float
    target_bpm: float
    rate_a: float
    rate_b: float
    note: str = ""
    corrections: list[str] = Field(default_factory=list)

    @property
    def was_corrected(self) -> bool:
        return bool(self.corrections)


def clamp_rate(rate: float) -> tuple[float, str | None]:
    """Clamp a playback rate into the varispeed window.

    Returned outside the models so the mixer, the tools, and the guardrail all
    agree on one definition of "playable".
    """
    if rate < RATE_MIN:
        return RATE_MIN, f"rate {rate:.4f} raised to the {RATE_MIN} floor (+/-8% varispeed clamp)"
    if rate > RATE_MAX:
        return RATE_MAX, f"rate {rate:.4f} lowered to the {RATE_MAX} ceiling (+/-8% varispeed clamp)"
    return rate, None


def bpm_within_varispeed(track_bpm: float | None, target_bpm: float) -> bool:
    if not track_bpm or track_bpm <= 0:
        return False
    rate = target_bpm / track_bpm
    return RATE_MIN <= rate <= RATE_MAX


def playable_window(target_bpm: float) -> tuple[float, float]:
    """The BPM range a target tempo can actually beatmatch."""
    return target_bpm * RATE_MIN, target_bpm * RATE_MAX


def resolve_target_bpm(
    requested: float, bpm_a: float | None, bpm_b: float
) -> tuple[float, float, float, list[str]]:
    """The tempo both decks can play, and the rate each needs to get there.

    Solved directly rather than by clamping each rate in turn: clamping deck A
    moves the target, which can push deck B back out of its own window, and an
    unrolled correction pass leaves the two decks at different tempos while
    reporting success.

    Returns (target_bpm, rate_a, rate_b, corrections).
    """
    corrections: list[str] = []
    lo, hi = bpm_b * RATE_MIN, bpm_b * RATE_MAX
    if bpm_a:
        lo, hi = max(lo, bpm_a * RATE_MIN), min(hi, bpm_a * RATE_MAX)

    if lo > hi:
        # The two tempos are more than the window's width apart, so no common
        # tempo exists. Keep deck B playable and say plainly what deck A will do.
        target = min(max(requested, bpm_b * RATE_MIN), bpm_b * RATE_MAX)
        rate_a, _ = clamp_rate(target / bpm_a)
        corrections.append(
            f"{bpm_a:.1f} and {bpm_b:.1f} BPM are further apart than the "
            f"+/-8% varispeed window allows; deck A can only reach "
            f"{bpm_a * rate_a:.1f} BPM against deck B's {target:.1f} -- "
            "these two will not beat-match"
        )
        return target, rate_a, target / bpm_b, corrections

    target = min(max(requested, lo), hi)
    if abs(target - requested) > 0.01:
        corrections.append(
            f"target {requested:.1f} -> {target:.1f} BPM "
            f"(+/-8% varispeed window around {'both decks' if bpm_a else 'deck B'})"
        )
    # target is inside both windows by construction, so any excursion here is
    # float noise from the division; snapping keeps the reported rates in range.
    def _snap(rate: float) -> float:
        return min(max(rate, RATE_MIN), RATE_MAX)

    return target, (_snap(target / bpm_a) if bpm_a else 1.0), _snap(target / bpm_b), corrections


def validate_plan(raw: dict, record: Analysis, bpm_a: float | None) -> ValidatedPlan:
    """Clamp a proposed transition into something the mixer can actually do.

    The planner model is advisory. This is authoritative for the crossfade
    length, the cue point, and the tempo both decks are asked to play; `rate_a`
    and `rate_b` are reported for the trace but the mixer only ramps deck A.
    """
    plan = TransitionPlan.model_validate(raw)

    bars = max(1, min(32, plan.crossfade_bars))
    corrections = (
        [f"crossfade_bars {plan.crossfade_bars} -> {bars}"] if bars != plan.crossfade_bars else []
    )

    target, rate_a, rate_b, tempo_notes = resolve_target_bpm(
        plan.target_bpm, bpm_a, record.bpm_used
    )
    corrections += tempo_notes

    # The crossfade has to fit before the end of the incoming track, with a few
    # seconds spare. Same arithmetic as `mixer.bars_to_samples`, in seconds.
    crossfade_s = bars * BEATS_PER_BAR * 60.0 / target
    max_cue = max(0.0, record.duration_s - crossfade_s - 5.0)
    cue = plan.cue_seconds
    if cue > max_cue:
        corrections.append(f"cue_seconds {cue:.1f} -> {max_cue:.1f} (too close to the end)")
        cue = max_cue

    return ValidatedPlan(
        crossfade_bars=bars,
        cue_seconds=round(cue, 2),
        target_bpm=round(target, 2),
        rate_a=round(rate_a, 4),
        rate_b=round(rate_b, 4),
        note=plan.note,
        corrections=corrections,
    )
