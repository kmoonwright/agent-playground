"""The brains: one protocol, three implementations.

`RuleBrain` needs no API key and calls exactly the same `DJSession` tool
implementations as the LLM brains, through the same `tools.Dispatch`. That is
what makes the audio pipeline testable and demoable with zero credentials, and
it doubles as the autopilot fallback when a sub-agent does not answer in time.

`LlmBrain` is the host agent; it delegates to the sub-agents in `subagents.py`.
"""

from __future__ import annotations

import re
import threading
from queue import Empty
from typing import Callable

import config
import instrumentation as ins
import agents.tools as tools
from agents.llm import LlmBrain
from session import DJSession
from music.source import Track

Say = Callable[[str], None]

BPM_RE = re.compile(r"(\d{2,3})\s*(?:bpm|beats)", re.I)
FASTER = re.compile(r"\b(upbeat|faster|harder|more energy|energetic|pump|lift|banger)\b", re.I)
SLOWER = re.compile(r"\b(chill|slower|downtempo|mellow|calm|relax|wind down|softer)\b", re.I)
SKIP_RE = re.compile(r"\b(skip|next track|next one|next|move on|change it)\b", re.I)
STATUS_RE = re.compile(r"\b(what'?s playing|now playing|status|what is this|what track)\b", re.I)

# Genre / instrumentation words the rule selector can turn into source tags.
TAG_WORDS = (
    "vocals",
    "instrumental",
    "hip_hop",
    "dub",
    "house",
    "ambient",
    "funk",
    "jazz",
    "techno",
    "downtempo",
    "guitar",
    "piano",
    "electronic",
    "acoustic",
)
TAG_ALIASES = {"hip hop": "hip_hop", "hiphop": "hip_hop", "vocal": "vocals", "singing": "vocals"}

STEP_BPM = 4.0


def extract_tags(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for alias, tag in TAG_ALIASES.items():
        if alias in lowered and tag not in found:
            found.append(tag)
    for tag in TAG_WORDS:
        if tag.replace("_", " ") in lowered or tag in lowered:
            if tag not in found:
                found.append(tag)
    return found


class RuleBrain:
    """Deterministic regex selector. No network beyond the sources themselves."""

    name = "rule"

    def __init__(self, session: DJSession, say: Say) -> None:
        self.session = session
        self.say = say
        self.dispatch = tools.Dispatch(session, "host", delegate=self._delegate)

    # -------------------------------------------------------- Brain

    def handle(self, utterance: str) -> None:
        with ins.using_set_context(self.session.set_id, {"brain": self.name}):
            with ins.traced_span("dj.turn", ins.CHAIN, input_value=utterance) as turn:
                with ins.traced_span(
                    "rule_selector",
                    ins.AGENT,
                    attributes={"role": "host", "brain": "rule", "fallback": "rule"},
                ):
                    reply = self._route(utterance)
                self.say(f"[dj] {reply}")
                ins.set_output(turn, reply)

    # -------------------------------------------------------- routing

    def _route(self, utterance: str) -> str:
        text = utterance.strip()

        if text.startswith("[autopilot]"):
            return self._pick_and_queue("", reason="autopilot")

        if STATUS_RE.search(text):
            return self.dispatch.run("get_now_playing", {})

        match = BPM_RE.search(text)
        if match:
            return self.dispatch.run("set_target_bpm", {"bpm": float(match.group(1))})

        if FASTER.search(text):
            return self.dispatch.run(
                "set_target_bpm", {"bpm": self.session.target_bpm() + STEP_BPM}
            )
        if SLOWER.search(text):
            return self.dispatch.run(
                "set_target_bpm", {"bpm": self.session.target_bpm() - STEP_BPM}
            )

        if SKIP_RE.search(text):
            result = self.dispatch.run("skip", {"crossfade_bars": 2})
            if "Nothing is queued" in result:
                queued = self._pick_and_queue(text, reason="skip")
                return f"{result} {queued}"
            return result

        return self._pick_and_queue(text, reason="request")

    def _delegate(self, name: str, args: dict) -> str:
        """RuleBrain has no sub-agents; do the deterministic equivalent instead."""
        if name == "find_tracks":
            lo, hi = self.session.playable_window()
            tracks = self.session.find_candidates(
                args.get("bpm_min") or lo, args.get("bpm_max") or hi, limit=6
            )
            if not tracks:
                return "No candidates found."
            return "; ".join(f"{t.track_id} {t.label}" for t in tracks[:3])
        return "Using the default 8-bar blend (no planner in rule mode)."

    # -------------------------------------------------------- selection

    def _pick_and_queue(self, text: str, *, reason: str) -> str:
        tags = extract_tags(text)
        target = self.session.target_bpm()
        lo, hi = self.session.playable_window()
        candidates = self.session.find_candidates(lo, hi, tags=tags, limit=6)
        if not candidates and tags:
            self.say(f"[dj] nothing tagged {', '.join(tags)} in range; widening the search")
            candidates = self.session.find_candidates(lo, hi, limit=6)
        if not candidates:
            return (
                f"I could not find anything playable near {target:.0f} BPM "
                f"(the +/-8% window is {lo:.0f}-{hi:.0f})."
            )

        already = self.session.spoken_for()

        queued = self._try_queue(candidates, already, tags)
        if queued:
            return queued

        # Every tagged match is already playing or queued -- widen rather than
        # give up, which is what a DJ would do.
        if tags:
            wider = self.session.find_candidates(lo, hi, limit=6)
            queued = self._try_queue(wider, already, [])
            if queued:
                return (
                    f"Nothing new tagged {', '.join(tags)} was available, so I widened it. "
                    f"{queued}"
                )
        return "Everything I can reach near this tempo is already playing or queued."

    def _try_queue(
        self, candidates: list[Track], already: set[str], tags: list[str]
    ) -> str | None:
        for track in candidates:
            if track.track_id in already:
                continue
            result = self.dispatch.run(
                "queue_track", {"track_id": track.track_id, "crossfade_bars": 8}
            )
            if result.startswith("Queued"):
                tag_note = f" (tags: {', '.join(tags)})" if tags else ""
                return f"{result}{tag_note}"
        return None


# ---------------------------------------------------------------- runner


class BrainRunner:
    """Owns the brain thread. Pulls utterances off the session's queue."""

    def __init__(self, brain: RuleBrain | LlmBrain, session: DJSession, say: Say) -> None:
        self.brain = brain
        self.session = session
        self.say = say
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.busy = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="brain", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self.session.q_user.get(timeout=0.25)
            except Empty:
                continue
            self.busy.set()
            try:
                self.brain.handle(utterance)
            except Exception as exc:
                self.say(f"[brain] {type(exc).__name__}: {exc}")
            finally:
                self.busy.clear()


def make_brain(provider: str, session: DJSession, say: Say) -> RuleBrain | LlmBrain:
    if provider == "rule":
        say("[brain] rule selector (no API key needed)")
        return RuleBrain(session, say)

    models = config.resolved_models(provider)
    say(f"[brain] {provider} — host on {models['host']}")
    return LlmBrain(provider, session, say)
