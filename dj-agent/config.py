"""Configuration, paths, audio constants, and the per-role model table.

This is the only module that reads the environment or decides which model a
role runs on. It deliberately does not import numpy, the LLM SDKs, or anything
from `instrumentation` — every other module imports config, so config must stay
at the bottom of the dependency graph.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

# Scoped to this project's own directory, not the cwd — same as the sibling
# demos, so running `python ../dj-agent/dj.py` still picks up the right .env.
# override=True is load-bearing in this repo: a shell that previously ran
# pdf-agent will still have ARIZE_PROJECT_NAME=pdf-extraction-demo, and the
# default (override=False) would keep sending DJ traces to that project.
load_dotenv(ROOT / ".env", override=True)

CACHE_DIR = ROOT / "cache"
TRACKS_DIR = CACHE_DIR / "tracks"
API_CACHE_DIR = CACHE_DIR / "api"
CREDITS_PATH = CACHE_DIR / "credits.jsonl"
OUT_DIR = ROOT / "out"
DEFAULT_CRATE_DIR = ROOT / "data" / "crate"

# ---------------------------------------------------------------- audio

SR = 44100
BLOCKSIZE = 1024
CHANNELS = 2

# Varispeed beatmatching shifts pitch with tempo, so the playback rate is
# clamped. ±8% is roughly what a turntable pitch fader gives you and is the
# hard constraint that bounds which tracks may be queued against a target BPM.
RATE_MIN = 0.92
RATE_MAX = 1.08

DEFAULT_TARGET_BPM = 122.0
DEFAULT_CROSSFADE_BARS = 8
BEATS_PER_BAR = 4

# Loudness normalization target at load time, so transitions don't jump level.
TARGET_RMS_DBFS = -14.0
PEAK_CEILING = 0.95

# Prefetch / scheduling (seconds of deck A remaining).
PREFETCH_LEAD_S = 90.0
AUTOPILOT_LEAD_S = 45.0
LOAD_DEADLINE_SLACK_S = 8.0

# Source constraints — skip anything too big or too long to be a demo track.
MAX_TRACK_BYTES = 20 * 1024 * 1024
MAX_TRACK_SECONDS = 8 * 60

# Span attribute truncation.
MAX_ATTR_CHARS = 4000

# Titles, artists and tags are attacker-controlled -- anyone can upload to
# ccMixter, and a crate is whatever files are in a folder. Escape sequences in
# one of those could rewrite the terminal scrollback, including forging the
# "license verified" line, which is the only honesty mechanism the setlist has.
MAX_LINE_CHARS = 400
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Strip terminal control characters and cap each line. Tabs and \\n survive."""
    lines = _CONTROL_CHARS.sub("", text).split("\n")
    return "\n".join(
        ln if len(ln) <= MAX_LINE_CHARS else ln[:MAX_LINE_CHARS] + "..." for ln in lines
    )


# ---------------------------------------------------------------- agents

ROLES = ("host", "crate_digger", "transition_planner")

# The expensive model hosts the conversation; the sub-agents that do the
# reading and the arithmetic run on something cheap.
DEFAULT_MODELS: dict[tuple[str, str], str] = {
    ("claude", "host"): "claude-opus-5",
    ("claude", "crate_digger"): "claude-haiku-4-5",
    ("claude", "transition_planner"): "claude-haiku-4-5",
    ("openai", "host"): "gpt-4o-mini",
    ("openai", "crate_digger"): "gpt-4o-mini",
    ("openai", "transition_planner"): "gpt-4o-mini",
}

MAX_ITERATIONS = {"host": 6, "crate_digger": 5, "transition_planner": 3}

# Populated by dj.py from CLI flags before any brain is constructed.
_model_overrides: dict[str, str] = {}
_model_override_all: str | None = None


def set_model_overrides(per_role: dict[str, str | None], every_role: str | None) -> None:
    """Record --model-<role> / --model CLI values. Call once, from main()."""
    global _model_override_all
    _model_overrides.clear()
    for role, value in per_role.items():
        if value:
            _model_overrides[role] = value
    _model_override_all = every_role


def resolve_model(provider: str, role: str) -> str:
    """CLI per-role > CLI all > env per-role > env all > built-in table."""
    if role in _model_overrides:
        return _model_overrides[role]
    if _model_override_all:
        return _model_override_all
    env_role = os.environ.get(f"DJ_MODEL_{role.upper()}")
    if env_role:
        return env_role
    env_all = os.environ.get("DJ_MODEL")
    if env_all:
        return env_all
    try:
        return DEFAULT_MODELS[(provider, role)]
    except KeyError:
        raise SystemExit(
            f"No default model for provider={provider!r} role={role!r}. "
            f"Pass --model-{role.replace('_', '-')} or set DJ_MODEL_{role.upper()}."
        )


def resolved_models(provider: str) -> dict[str, str]:
    return {role: resolve_model(provider, role) for role in ROLES}


# ---------------------------------------------------------------- accessors

def _env(name: str) -> str | None:
    return (os.environ.get(name) or "").strip() or None


def arize_api_key() -> str | None:
    return _env("ARIZE_API_KEY")


def arize_space_id() -> str | None:
    return _env("ARIZE_SPACE_ID")


def arize_project_name() -> str:
    return _env("ARIZE_PROJECT_NAME") or "dj-agent"


def anthropic_api_key() -> str | None:
    return _env("ANTHROPIC_API_KEY")


def openai_api_key() -> str | None:
    return _env("OPENAI_API_KEY")


def auto_provider() -> str:
    """Pick a provider from whatever credentials are present."""
    if anthropic_api_key():
        return "claude"
    if openai_api_key():
        return "openai"
    return "rule"


def require_provider(provider: str) -> str:
    """Resolve `auto` and fail loudly if an explicit provider has no key."""
    if provider == "auto":
        return auto_provider()
    if provider == "claude" and not anthropic_api_key():
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or run with --provider rule for the keyless selector."
        )
    if provider == "openai" and not openai_api_key():
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or run with --provider rule for the keyless selector."
        )
    return provider


def ensure_dirs() -> None:
    for path in (CACHE_DIR, TRACKS_DIR, API_CACHE_DIR, OUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
