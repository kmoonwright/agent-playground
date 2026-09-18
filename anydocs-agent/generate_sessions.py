"""Build multi-turn conversations, grouped into Arize AX sessions.

Each session is one persona walking a 4-turn AX workflow (instrumentation,
evals, reliability, MCP, compound retrieval, or a coverage-boundary
refusal). All 4 turns share one session.id so AX's Sessions view shows a
real conversation instead of 4 unrelated traces. The questions are the
ones in EXAMPLE_PROMPTS.md plus the eval dataset — already known to
produce the interesting span trees (1-hop, 2-hop, MCP client/server,
guardrail refusal).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
import textwrap

import config
import instrumentation
import mcp_client
from agents import host

# Each journey is a customer-demo conversation: 4 self-contained questions
# that still read as one person's workflow in AX Sessions. Names land in
# session.id so you can pick them out of the list without opening a trace.
JOURNEYS: list[dict] = [
    {
        "name": "span-onboarding",
        "messages": [
            "What are the eleven OpenInference span kinds?",
            "What's the difference between EMBEDDING and RETRIEVER spans?",
            "What function sets up Arize tracing, and what do you do with what it returns?",
            "Why does tracing setup have to run before the instrumented SDK is imported?",
        ],
    },
    {
        "name": "eval-lifecycle",
        "messages": [
            "Who orchestrates an online evaluator versus an offline evaluator on Arize AX?",
            "What's the difference between code-based evals and LLM-as-a-judge evals?",
            "How could an Arize AX online evaluator implement the groundedness metric described in the LLM evaluation guide?",
            "How do online and offline evaluators compose in a typical setup?",
        ],
    },
    {
        "name": "reliability-loop",
        "messages": [
            "Name the five silent agent failure modes described for production reliability.",
            "What are the five silent failure modes, and what does the eight-step improvement loop look like?",
            "What does a reliability scorecard track?",
            "Why do long trajectories compound reliability risk?",
        ],
    },
    {
        "name": "mcp-tracing",
        "messages": [
            "Does MCPInstrumentor create its own spans when tracing an MCP call?",
            "Why can't an MCP server using stdio transport print debug output to stdout?",
            "What OpenTelemetry context actually crosses the MCP wire, and what doesn't?",
            "What do you still have to do yourself when using MCPInstrumentor?",
        ],
    },
    {
        "name": "compound-deep",
        "messages": [
            "What's the difference between EMBEDDING and RETRIEVER spans, and how does online vs offline evaluation relate to that?",
            "What are the five silent failure modes, and what does the eight-step improvement loop look like?",
            "How could an Arize AX online evaluator implement the groundedness metric described in the LLM evaluation guide?",
            "Does MCPInstrumentor create its own spans, and why would that matter for naming mcp_client vs mcp_server spans?",
        ],
    },
    {
        "name": "coverage-boundary",
        "messages": [
            "What are the eleven OpenInference span kinds?",
            "What is Arize's Enterprise pricing?",
            "What's the difference between code-based evals and LLM-as-a-judge evals?",
            "Why can't an MCP server using stdio transport print debug output to stdout?",
        ],
    },
]


def build_session_plans(count: int) -> list[dict]:
    """count session plans, cycling journeys. Each: session_id, name, messages."""
    plans = []
    for idx, journey in enumerate(itertools.islice(itertools.cycle(JOURNEYS), count)):
        plans.append(
            {
                "session_id": f"session-{journey['name']}-{idx:02d}",
                "name": journey["name"],
                "messages": list(journey["messages"]),
            }
        )
    return plans


def _preview(text: str, width: int = 88) -> str:
    collapsed = " ".join(text.split())
    return textwrap.shorten(collapsed, width=width, placeholder="…")


async def run_session(plan: dict, docs_dir: str | None) -> list[dict]:
    """Run every turn of one session sequentially, sharing one session.id.

    Turns stay sequential on purpose: AX Sessions is a conversation, and
    each turn here is one trace under that id. One bad turn is recorded
    and skipped so the rest of the session still lands.
    """
    turns = []
    async with mcp_client.open_session(docs_dir) as session:
        for message in plan["messages"]:
            try:
                answer = await host.run(session, plan["session_id"], message)
                turns.append({"message": message, "answer": answer, "error": None})
            except Exception as exc:  # noqa: BLE001 -- one bad turn shouldn't kill the session
                turns.append({"message": message, "answer": None, "error": str(exc)})
    return turns


def _print_session(plan: dict, turns: list[dict]) -> None:
    print(f"\nsession_id={plan['session_id']} journey={plan['name']}")
    for turn in turns:
        if turn["error"] is not None:
            print(f"  FAILED  {_preview(turn['message'])}\n          {turn['error']}")
            continue
        print(f"  Q  {_preview(turn['message'])}")
        print(f"  A  {_preview(turn['answer'] or '')}")


async def _run_all(plans: list[dict], concurrency: int, docs_dir: str | None) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def one(plan: dict) -> tuple[dict, list[dict], Exception | None]:
        async with sem:
            try:
                return plan, await run_session(plan, docs_dir), None
            except Exception as exc:  # noqa: BLE001 -- one bad session shouldn't kill the batch
                return plan, [], exc

    tasks = [asyncio.create_task(one(plan)) for plan in plans]
    for finished in asyncio.as_completed(tasks):
        plan, turns, exc = await finished
        if exc is not None:
            print(f"session_id={plan['session_id']} FAILED: {exc}")
            continue
        _print_session(plan, turns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build multi-turn conversations, grouped into Arize AX sessions."
    )
    parser.add_argument("--sessions", type=int, default=len(JOURNEYS), help="Number of sessions to run.")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent sessions (each spawns an MCP server).")
    parser.add_argument(
        "--docs", help="path to a folder of markdown docs (default: the bundled Arize/OpenInference docs)"
    )
    args = parser.parse_args(argv)

    if not config.arize_api_key() or not config.arize_space_id():
        raise SystemExit(
            "ARIZE_API_KEY and ARIZE_SPACE_ID are required to export demo sessions "
            "to Arize AX. Copy .env.example to .env and fill them in from https://app.arize.com."
        )
    if not config.openai_api_key():
        print(
            "Warning: OPENAI_API_KEY is not set. Sessions will still export, "
            "but every turn takes exactly one hop (the mock never multi-hops).",
            file=sys.stderr,
        )

    instrumentation.ensure_tracing()
    plans = build_session_plans(args.sessions)
    try:
        asyncio.run(_run_all(plans, args.concurrency, args.docs))
    finally:
        instrumentation.flush_tracing()

    print(
        f"\n{len(plans)} sessions run. Check Arize AX -> project "
        f"'{config.arize_project_name()}' -> Sessions."
    )
    for plan in plans:
        print(f"  {plan['session_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
