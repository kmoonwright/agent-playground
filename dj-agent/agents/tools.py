"""Tool specs, written once, provider-neutral, plus the dispatch into DJSession.

A spec is `{name, description, input_schema}` — which is already the Anthropic
shape, so `to_anthropic()` is close to a pass-through and `to_openai()` wraps
each one as a function tool with `parameters`. Writing them once is what makes
`--provider claude` and `--provider openai` behave the same, and what lets a
single pytest assert both adaptations from one source of truth.

Tool descriptions state *when* to call, not just what the tool does. That is
deliberate: recent models are conservative about reaching for tools, and a
trigger condition in the description is the highest-leverage place to fix it.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from config import RATE_MAX, RATE_MIN
from schema import BPM_ARG_MAX, BPM_ARG_MIN, Candidates

Spec = dict[str, Any]

# One definition of the BPM bounds, matching `schema.TempoRange`.
_bpm = {"type": "integer", "minimum": BPM_ARG_MIN, "maximum": BPM_ARG_MAX}


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ---------------------------------------------------------------- host tools

GET_NOW_PLAYING: Spec = {
    "name": "get_now_playing",
    "description": (
        "Report the state of the decks: what is playing, its source BPM and current playback "
        "rate, how much time is left, whether deck B is staged, the target BPM, and the queue. "
        "Call this whenever the user asks what is playing, or before deciding on a transition, "
        "since how much time is left determines how long a crossfade can be."
    ),
    "input_schema": _obj({}, []),
}

SET_TARGET_BPM: Spec = {
    "name": "set_target_bpm",
    "description": (
        "Move the tempo of the set. Ramps the currently playing track's playback rate over "
        f"`ramp_bars` bars. Playback rate is clamped to {RATE_MIN}-{RATE_MAX} (+/-8%), because "
        "beatmatching here is varispeed: tempo and pitch move together, like a turntable pitch "
        "fader. A request beyond that window is clamped and you are told so. Call this when the "
        "user asks for a different tempo, or wants the energy up or down."
    ),
    "input_schema": _obj(
        {
            "bpm": {
                "type": "number",
                "minimum": 60,
                "maximum": 180,
                "description": "The tempo the set should move to, in BPM.",
            },
            "ramp_bars": {
                "type": "integer",
                "minimum": 1,
                "maximum": 32,
                "description": "How many bars to spread the tempo change over. 4 is musical.",
            },
        },
        ["bpm"],
    ),
}

SKIP: Spec = {
    "name": "skip",
    "description": (
        "Start the next transition immediately, on the next bar, instead of waiting for the "
        "current track to run out. Requires something already queued and staged. Call this when "
        "the user says skip, next, or that they are done with the current track."
    ),
    "input_schema": _obj(
        {
            "crossfade_bars": {
                "type": "integer",
                "minimum": 1,
                "maximum": 32,
                "description": "Length of the crossfade. Short (2-4) for an impatient skip.",
            }
        },
        [],
    ),
}

QUEUE_TRACK: Spec = {
    "name": "queue_track",
    "description": (
        "Queue a track by the id you got from find_tracks. The track is downloaded and analysed "
        "in the background; the transition happens automatically before the current track runs "
        "out. Rejected if the track's tempo is too far from the target to beatmatch. Call this "
        "once you have chosen a track — nothing plays until you do."
    ),
    "input_schema": _obj(
        {
            "track_id": {"type": "string", "description": "id from find_tracks."},
            "crossfade_bars": {
                "type": "integer",
                "minimum": 1,
                "maximum": 32,
                "description": "8 is a normal blend; 2-4 is a quick cut; 16+ is a long ride.",
            },
            "cue_seconds": {
                "type": "number",
                "minimum": 0,
                "description": "Skip this far into the track before it enters. Use to skip an intro.",
            },
            "position": {
                "type": "string",
                "enum": ["next", "end"],
                "description": "Where in the queue to put it.",
            },
        },
        ["track_id"],
    ),
}

FIND_TRACKS: Spec = {
    "name": "find_tracks",
    "description": (
        "Hand a brief to the crate digger, which searches every enabled source, inspects "
        "candidates, and returns a short ranked shortlist with reasons. Give it a tempo range "
        "that is within +/-8% of the current target BPM, or nothing it finds will be playable. "
        "Call this whenever you need a track you do not already have an id for. You may call it "
        "twice in one turn to compare two different directions."
    ),
    "input_schema": _obj(
        {
            "brief": {
                "type": "string",
                "description": (
                    "What you want, in words: genre, mood, instrumentation, energy. "
                    "e.g. 'warm dubby house with vocals, not too busy'."
                ),
            },
            "bpm_min": _bpm,
            "bpm_max": _bpm,
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "How many candidates to return.",
            },
        },
        ["brief", "bpm_min", "bpm_max"],
    ),
}

PLAN_TRANSITION: Spec = {
    "name": "plan_transition",
    "description": (
        "Hand a chosen track to the transition planner, which reads both tracks' analyses and "
        "returns a validated crossfade length, cue point, and target BPM. The plan is clamped to "
        "what the mixer can actually do before you see it. Call this when you want a considered "
        "transition rather than the default 8-bar blend — a long ride, a quick cut, or skipping "
        "a long intro."
    ),
    "input_schema": _obj(
        {
            "to_track_id": {"type": "string", "description": "The track you intend to queue."},
            "style_hint": {
                "type": "string",
                "description": "e.g. 'long blend', 'quick cut', 'skip the ambient intro'.",
            },
        },
        ["to_track_id"],
    ),
}

HOST_TOOLS: list[Spec] = [
    GET_NOW_PLAYING,
    SET_TARGET_BPM,
    SKIP,
    QUEUE_TRACK,
    FIND_TRACKS,
    PLAN_TRANSITION,
]


# ---------------------------------------------------------------- crate_digger

SEARCH_SOURCE: Spec = {
    "name": "search_source",
    "description": (
        "Search one source for tracks in a tempo range. Returns compact records including "
        "whether each track is already cached (instant to play) and its grid confidence (how "
        "much its beat grid can be trusted). Call this repeatedly: different sources, different "
        "tempo bands, different tags. Prefer cached tracks and high grid confidence."
    ),
    "input_schema": _obj(
        {
            "source": {"type": "string", "description": "An enabled source name."},
            "bpm_min": _bpm,
            "bpm_max": _bpm,
            "tags": {
                "type": "string",
                "description": "Comma-separated tags, e.g. 'vocals,house'. Optional.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 12},
        },
        ["source", "bpm_min", "bpm_max"],
    ),
}

INSPECT_TRACK: Spec = {
    "name": "inspect_track",
    "description": (
        "Read everything known about one track: license, tags, tempo, how much the beat grid is "
        "trusted. Call this on a shortlist before reporting, especially when two candidates look "
        "equally good on tempo."
    ),
    "input_schema": _obj({"track_id": {"type": "string"}}, ["track_id"]),
}

REPORT_CANDIDATES: Spec = {
    "name": "report",
    "description": (
        "Submit your ranked shortlist and finish. Call this exactly once, at the end. Give one "
        "short concrete reason per candidate — the DJ reads these to choose, so 'matches the "
        "122 BPM target and has the vocal the user asked for' beats 'good fit'."
    ),
    "input_schema": _obj(
        {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": _obj(
                    {
                        "track_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    ["track_id", "reason"],
                ),
            }
        },
        ["candidates"],
    ),
}

DIGGER_TOOLS: list[Spec] = [SEARCH_SOURCE, INSPECT_TRACK, REPORT_CANDIDATES]


# ---------------------------------------------------------------- planner

GET_ANALYSIS: Spec = {
    "name": "get_analysis",
    "description": (
        "Read deck A's live state and the candidate track's analysis together: both tempos, how "
        "much of deck A is left, the target BPM, and the playback rate the candidate would need. "
        "Call this first — you cannot plan a transition without it."
    ),
    "input_schema": _obj({"track_id": {"type": "string"}}, ["track_id"]),
}

SUBMIT_PLAN: Spec = {
    "name": "submit_plan",
    "description": (
        "Submit the transition and finish. Call this exactly once, at the end. Your numbers are "
        "advisory: a validator clamps them to the varispeed window and the track's length before "
        "the mixer sees them, and tells the DJ what it had to change."
    ),
    "input_schema": _obj(
        {
            "crossfade_bars": {
                "type": "integer",
                "minimum": 1,
                "maximum": 32,
                "description": "8 is a normal blend, 2-4 a cut, 16+ a long ride.",
            },
            "cue_seconds": {
                "type": "number",
                "minimum": 0,
                "description": "How far into the incoming track to start. Use to skip an intro.",
            },
            "target_bpm": {
                "type": "number",
                "minimum": 60,
                "maximum": 180,
                "description": "The tempo both decks should run at through the blend.",
            },
            "note": {
                "type": "string",
                "description": "One line on the musical reasoning, shown to the user.",
            },
        },
        ["crossfade_bars", "cue_seconds", "target_bpm"],
    ),
}

PLANNER_TOOLS: list[Spec] = [GET_ANALYSIS, SUBMIT_PLAN]

TOOLS_BY_ROLE: dict[str, list[Spec]] = {
    "host": HOST_TOOLS,
    "crate_digger": DIGGER_TOOLS,
    "transition_planner": PLANNER_TOOLS,
}

TERMINAL_TOOL: dict[str, str] = {
    "crate_digger": "report",
    "transition_planner": "submit_plan",
}


# ---------------------------------------------------------------- adapters


def to_anthropic(specs: list[Spec]) -> list[dict[str, Any]]:
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["input_schema"],
        }
        for s in specs
    ]


def to_openai(specs: list[Spec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in specs
    ]


def names(specs: list[Spec]) -> list[str]:
    return [s["name"] for s in specs]


# ---------------------------------------------------------------- dispatch


class Dispatch:
    """Maps a tool name to a DJSession call, per role.

    Every handler returns a string observation. Sub-agent terminal tools also
    stash their parsed result in `self.terminal`, which is how the sub-agent
    loop knows it is finished.
    """

    def __init__(self, session, role: str, *, delegate: Callable[..., str] | None = None) -> None:
        self.session = session
        self.role = role
        self.delegate = delegate
        self.terminal: Any = None
        self.calls: list[str] = []

    def run(self, name: str, args: dict[str, Any]) -> str:
        allowed = names(TOOLS_BY_ROLE[self.role])
        if name not in allowed:
            return f"Unknown tool {name!r}. Available to you: {', '.join(allowed)}."
        # Recorded after the check, so a hallucinated name never lands in the
        # tool_calls span attribute beside the real ones.
        self.calls.append(name)

        try:
            if name == "get_now_playing":
                return self.session.tool_get_now_playing()
            if name == "set_target_bpm":
                return self.session.tool_set_target_bpm(args)
            if name == "skip":
                return self.session.tool_skip(args)
            if name == "queue_track":
                return self.session.tool_queue_track(args)
            if name == "search_source":
                return self.session.tool_search_source(args)
            if name == "inspect_track":
                return self.session.tool_inspect_track(args)
            if name == "get_analysis":
                return self.session.tool_get_analysis(args)

            if name == "report":
                parsed = Candidates.model_validate(args)
                self.terminal = parsed
                return f"Reported {len(parsed.candidates)} candidates."

            if name == "submit_plan":
                self.terminal = args
                return "Plan submitted."

            if name in {"find_tracks", "plan_transition"}:
                return self.delegate(name, args)
        except ValidationError as exc:
            return f"Invalid arguments for {name}: {_short_validation(exc)}"
        except Exception as exc:
            return f"{name} failed: {type(exc).__name__}: {exc}"

        return f"Tool {name!r} is not implemented."


def _short_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:4]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)
