"""Fire a batch of agent turns quickly to populate an Arize AX tenant.

Not the dataset/experiment path (see run_experiment.py) -- no Dataset or
Experiment object, just raw agent.run_agent() calls so each one lands as its
own trace. Useful for seeding a tenant with a good mix of trace shapes right
before a live demo.
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import agent
import config
import dataset_gen
import tracing


def build_pool(*, compound_only: bool = False, guardrail_only: bool = False) -> list[dict]:
    """Rows to draw from, cycled to reach --count. Each has user_id/message/message_history.

    Default (no flags) already includes every category -- plain FAQ/report/
    dispute rows, multi-tool compound rows, and guardrail-triggering rows --
    so one plain run surfaces the full range of trace shapes.
    """
    if compound_only:
        return dataset_gen.compound_rows()
    if guardrail_only:
        return dataset_gen.guardrail_rows()
    return dataset_gen.generate_rows() + dataset_gen.compound_rows() + dataset_gen.guardrail_rows()


def select(pool: list[dict], count: int) -> list[dict]:
    return list(itertools.islice(itertools.cycle(pool), count))


def _run_one(row: dict) -> tuple[dict, agent.AgentResult | None, Exception | None]:
    try:
        result = agent.run_agent(row["user_id"], row["message"], row.get("message_history"))
        return row, result, None
    except Exception as exc:  # noqa: BLE001 -- one bad call shouldn't kill the batch
        return row, None, exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fire a batch of agent turns to populate Arize AX with demo traces."
    )
    parser.add_argument("--count", type=int, default=30, help="Total number of agent turns to run.")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent agent turns.")
    parser.add_argument(
        "--compound-only",
        action="store_true",
        help="Only fire multi-tool COMPOUND_PROMPTS, for fast, deep traces.",
    )
    parser.add_argument(
        "--guardrail-only",
        action="store_true",
        help="Only fire GUARDRAIL_PROMPTS, to demo the input/output guardrails catching something.",
    )
    args = parser.parse_args()

    config.require_arize()
    tracing.ensure_tracing()

    pool = build_pool(compound_only=args.compound_only, guardrail_only=args.guardrail_only)
    rows = select(pool, args.count)

    ok = 0
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool_executor:
            futures = [pool_executor.submit(_run_one, row) for row in rows]
            for i, future in enumerate(as_completed(futures), start=1):
                row, result, exc = future.result()
                if exc is not None:
                    failed += 1
                    print(f"[{i}/{len(rows)}] user_id={row['user_id']} FAILED: {exc}")
                    continue
                ok += 1
                tool_names = [tc["name"] for tc in result.tool_calls]
                print(
                    f"[{i}/{len(rows)}] user_id={row['user_id']} tool_calls={tool_names} "
                    f"guardrails_triggered={result.guardrails_triggered} trace_id={result.trace_id}"
                )
    finally:
        tracing.flush_tracing()

    print(f"\n{ok} ok, {failed} failed. Check Arize AX -> project '{config.arize_project_name()}'.")


if __name__ == "__main__":
    main()
