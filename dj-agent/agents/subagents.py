"""The two sub-agents: crate_digger and transition_planner.

Each runs its own message loop, with its own system prompt, its own (cheaper)
model, and its own tool set. The host never sees raw search JSON or a raw plan —
only a ranked shortlist and a validated transition.

Both run on a small thread pool owned by the host, so the OpenTelemetry context
is re-attached by the caller via `instrumentation.in_thread`; without that these
AGENT spans would be trace roots instead of children of the host's tool call.

Neither sub-agent can touch the mixer, the queue, or the loader. They read the
library and return data; the host does the queueing. That keeps "who can mutate
playback" down to exactly one agent.
"""

from __future__ import annotations

from typing import Any

import config
import instrumentation as ins
import music.library as library
import agents.llm as llm
import agents.prompt as prompts
import agents.tools as tools
from schema import Candidates, FindTracksArgs, PlanTransitionArgs, playable_window


def _brief_for_digger(session, args: FindTracksArgs) -> str:
    target = session.target_bpm()
    enabled = ", ".join(sorted(session.sources))
    playable_lo, playable_hi = playable_window(target)
    return (
        f"Brief: {args.brief}\n"
        f"Tempo window requested: {args.bpm_min}-{args.bpm_max} BPM.\n"
        f"Return at most {args.limit} candidates.\n\n"
        f"The set is currently at {target:.1f} BPM, so anything between "
        f"{playable_lo:.0f} and {playable_hi:.0f} BPM is playable and anything outside that "
        f"is not, whatever the brief says.\n"
        f"Enabled sources: {enabled}."
    )


def run_crate_digger(provider: str, session, raw: dict[str, Any]) -> str:
    """Search, inspect, rank. Returns the shortlist as text for the host."""
    args = FindTracksArgs.model_validate(raw)
    model = config.resolve_model(provider, "crate_digger")

    with ins.traced_span(
        "crate_digger",
        ins.AGENT,
        input_value=args.brief,
        attributes={
            "role": "crate_digger",
            "model": model,
            "provider": provider,
            "bpm_min": args.bpm_min,
            "bpm_max": args.bpm_max,
            "limit": args.limit,
        },
    ) as span:
        dispatch = tools.Dispatch(session, "crate_digger")
        result = llm.run_agent(
            provider,
            role="crate_digger",
            system=prompts.DIGGER_SYSTEM,
            messages=[{"role": "user", "content": _brief_for_digger(session, args)}],
            dispatch=dispatch,
            max_iterations=config.MAX_ITERATIONS["crate_digger"],
        )

        terminal = result.terminal
        if isinstance(terminal, Candidates) and terminal.candidates:
            lines = []
            for candidate in terminal.candidates:
                track = library.load_track(candidate.track_id)
                label = track.label if track else candidate.track_id
                ref = library.to_ref(track, target_bpm=session.target_bpm()) if track else None
                bpm = f"{ref.bpm:.1f} BPM" if ref and ref.bpm else "unknown BPM"
                cached = " [cached]" if ref and ref.cached else ""
                lines.append(f"{candidate.track_id}  {label} — {bpm}{cached}\n    {candidate.reason}")
            ins.set_attributes(
                span,
                {
                    "candidates": len(terminal.candidates),
                    "iterations": result.iterations,
                    "tool_calls": result.tool_calls,
                    "fallback": "none",
                },
            )
            body = "\n".join(lines)
            ins.set_output(span, body)
            return f"{len(terminal.candidates)} candidate(s):\n{body}"

        # The digger did not report. Fall back to the deterministic search so the
        # host still gets something usable, and mark the span so it is obvious in
        # the trace which path ran.
        fallback = session.find_candidates(args.bpm_min, args.bpm_max, limit=args.limit)
        ins.set_attributes(
            span,
            {
                "candidates": len(fallback),
                "iterations": result.iterations,
                "tool_calls": result.tool_calls,
                "fallback": "rule",
                "refused": result.refused,
            },
        )
        if not fallback:
            ins.set_output(span, "no candidates")
            return (
                f"The crate digger found nothing playable between {args.bpm_min} and "
                f"{args.bpm_max} BPM. Try a wider tempo range."
            )
        body = "\n".join(
            f"{t.track_id}  {t.label} — {t.bpm:.1f} BPM" if t.bpm else f"{t.track_id}  {t.label}"
            for t in fallback
        )
        ins.set_output(span, body)
        return (
            "The crate digger did not report, so here is a plain tempo-matched search "
            f"instead:\n{body}"
        )


def run_transition_planner(provider: str, session, raw: dict[str, Any]) -> str:
    """Plan one transition, then let the guardrail have the last word."""
    args = PlanTransitionArgs.model_validate(raw)
    model = config.resolve_model(provider, "transition_planner")

    with ins.traced_span(
        "transition_planner",
        ins.AGENT,
        input_value=args.style_hint or args.to_track_id,
        attributes={
            "role": "transition_planner",
            "model": model,
            "provider": provider,
            "to_track_id": args.to_track_id,
        },
    ) as span:
        record = library.load_analysis(args.to_track_id)
        if record is None:
            ins.set_output(span, "not analysed")
            return (
                f"{args.to_track_id} has not been analysed yet, so there is nothing to plan "
                "against. Queue it with the default 8-bar blend, or wait for it to load."
            )

        dispatch = tools.Dispatch(session, "transition_planner")
        opening = (
            f"Plan the transition into {args.to_track_id}.\n"
            f"Style hint: {args.style_hint or '(none given)'}\n"
            "Call get_analysis first, then submit_plan."
        )
        result = llm.run_agent(
            provider,
            role="transition_planner",
            system=prompts.PLANNER_SYSTEM,
            messages=[{"role": "user", "content": opening}],
            dispatch=dispatch,
            max_iterations=config.MAX_ITERATIONS["transition_planner"],
        )

        if not isinstance(result.terminal, dict):
            ins.set_attributes(
                span,
                {
                    "iterations": result.iterations,
                    "tool_calls": result.tool_calls,
                    "fallback": "default_blend",
                },
            )
            ins.set_output(span, "no plan submitted")
            return (
                "The planner did not submit a plan. Use the default 8-bar blend: "
                f"queue_track with track_id={args.to_track_id} and crossfade_bars=8."
            )

        try:
            validated = session.validate_plan(result.terminal, args.to_track_id)
        except Exception as exc:
            ins.set_output(span, f"validation failed: {exc}")
            return f"The plan could not be validated ({exc}). Fall back to an 8-bar blend."

        session.remember_plan(args.to_track_id, validated)
        ins.set_attributes(
            span,
            {
                "iterations": result.iterations,
                "tool_calls": result.tool_calls,
                "fallback": "none",
                "corrected": validated.was_corrected,
                "crossfade_bars": validated.crossfade_bars,
                "cue_seconds": validated.cue_seconds,
                "target_bpm": validated.target_bpm,
            },
        )
        ins.set_output(span, validated.model_dump())

        lines = [
            f"Plan for {args.to_track_id}: {validated.crossfade_bars}-bar crossfade, "
            f"cue at {validated.cue_seconds:.1f}s, target {validated.target_bpm:.1f} BPM "
            f"(deck A rate {validated.rate_a:.4f}, deck B rate {validated.rate_b:.4f})."
        ]
        if validated.note:
            lines.append(validated.note)
        if validated.corrections:
            lines.append("The validator adjusted: " + "; ".join(validated.corrections) + ".")
        lines.append(
            f"Queue it with crossfade_bars={validated.crossfade_bars} and "
            f"cue_seconds={validated.cue_seconds:.1f}."
        )
        return " ".join(lines)
