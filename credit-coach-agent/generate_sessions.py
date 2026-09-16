"""Build realistic multi-turn conversations, grouped into Arize AX sessions.

Each session is one persona asking a fixed 4-step "prospect journey" --
general question, then report, then disputable items, then a specific
dispute -- with real accumulated history (the agent's actual prior replies,
not canned text). All 4 turns of one session share one session.id, so
Arize AX's Sessions view shows them as one longer-style conversation instead
of 4 unrelated traces. See generate_traces.py for single-turn, no-session
trace generation.
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import agent
import config
import db
import dataset_gen
import tracing

JOURNEY_CATEGORIES = ["how_it_works", "explain_report", "disputable_items", "can_i_dispute"]


def _dispute_item_id(user_id: str) -> str:
    items = db.list_disputable_items(user_id)["disputable_items"]
    return items[0]["item_id"] if items else "DI-99"


def session_messages(user_id: str) -> list[str]:
    """The 4 ordered messages for one persona's journey. Pure, no network."""
    return [
        dataset_gen.HOW_IT_WORKS_TEMPLATES[0],
        dataset_gen.EXPLAIN_REPORT_TEMPLATES[0],
        dataset_gen.DISPUTABLE_ITEMS_TEMPLATES[0],
        dataset_gen.CAN_DISPUTE_TEMPLATES[0].format(item_id=_dispute_item_id(user_id)),
    ]


def build_session_plans(count: int) -> list[dict]:
    """count session plans, cycling personas. Each: session_id, user_id, messages."""
    user_ids = db.list_user_ids()
    plans = []
    for idx, user_id in enumerate(itertools.islice(itertools.cycle(user_ids), count)):
        plans.append(
            {
                "session_id": f"session-{user_id}-{idx:02d}",
                "user_id": user_id,
                "messages": session_messages(user_id),
            }
        )
    return plans


def run_session(plan: dict) -> list[dict]:
    """Run all 4 turns of one session sequentially, sharing one session.id.

    History accumulates the agent's real replies, so later turns genuinely
    build on earlier ones.
    """
    turns = []
    history: list[dict] = []
    with tracing.using_session(plan["session_id"], extra={"user_id": plan["user_id"]}):
        for category, message in zip(JOURNEY_CATEGORIES, plan["messages"]):
            result = agent.run_agent(plan["user_id"], message, history)
            turns.append(
                {
                    "category": category,
                    "message": message,
                    "tool_calls": [tc["name"] for tc in result.tool_calls],
                    "trace_id": result.trace_id,
                }
            )
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": result.response_text})
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-turn conversations, grouped into Arize AX sessions."
    )
    parser.add_argument("--sessions", type=int, default=8, help="Number of sessions to run.")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent sessions.")
    args = parser.parse_args()

    config.require_arize()
    tracing.ensure_tracing()

    plans = build_session_plans(args.sessions)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_session, plan): plan for plan in plans}
            for future in as_completed(futures):
                plan = futures[future]
                try:
                    turns = future.result()
                except Exception as exc:  # noqa: BLE001 -- one bad session shouldn't kill the batch
                    print(f"session_id={plan['session_id']} FAILED: {exc}")
                    continue
                print(f"\nsession_id={plan['session_id']} user_id={plan['user_id']}")
                for turn in turns:
                    print(f"  [{turn['category']}] tool_calls={turn['tool_calls']} trace_id={turn['trace_id']}")
    finally:
        tracing.flush_tracing()

    print(f"\n{len(plans)} sessions run. Check Arize AX -> project '{config.arize_project_name()}' -> Sessions.")
    for plan in plans:
        print(f"  {plan['session_id']}")


if __name__ == "__main__":
    main()
