"""Offline eval: upload data/eval_dataset.json as an Arize dataset, run the
real answer pipeline as the task, score it with EVALUATORS, and upload the
results as an Arize experiment. This is the offline half of the online/
offline evaluator split — see data/docs/evaluators-online-vs-offline.md.
"""

import argparse
import json
import uuid

import config
import instrumentation
import mcp_client
from agents import host
from arize import ArizeClient
from evaluators import EVALUATORS

DATASET_NAME = "anydocs-agent-eval"
EXPERIMENT_NAME = "anydocs-agent-answer-quality"


async def _answer_question(question: str) -> dict:
    # Each row is an independent one-shot question, not a multi-turn
    # conversation, so each gets its own session_id.
    #
    # Each row also opens its own MCP server subprocess rather than sharing
    # one across the run: client.experiments.run() may run an async task
    # under a fresh event loop per call, and a stdio ClientSession is bound
    # to the loop that opened it — sharing one across loops isn't safe.
    # store.py's on-disk embedding cache is what keeps re-embedding the same
    # corpus every row cheap, not session reuse.
    async with mcp_client.open_session() as session:
        answer = await host.run(session, str(uuid.uuid4()), question)
    citations = sorted(host.extract_citations(answer))
    # Not an exact match against host.NOT_COVERED: the guardrail's forced
    # threshold uses that exact string, but the model's own "this isn't
    # covered" self-restraint (see ANSWER_SYSTEM) phrases it in its own
    # words. Citing nothing is the real signal either path produces.
    return {"answer": answer, "citations": citations, "refused": not citations}


async def task(dataset_row: dict) -> dict:
    return await _answer_question(dataset_row["question"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline eval experiment.")
    parser.add_argument("--dry-run", action="store_true", help="run locally without persisting the experiment")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args(argv)

    config.require_arize()
    instrumentation.ensure_tracing()

    try:
        client = ArizeClient(api_key=config.arize_api_key())
        rows = json.loads(config.EVAL_DATASET_PATH.read_text())
        # Dataset names must be unique per space — suffix so re-running this
        # script doesn't collide with the previous run's dataset.
        dataset_name = f"{DATASET_NAME}-{uuid.uuid4().hex[:8]}"
        dataset = client.datasets.create(space=config.arize_space_id(), name=dataset_name, examples=rows)
        _experiment, df = client.experiments.run(
            name=EXPERIMENT_NAME,
            dataset=dataset.id,
            space=config.arize_space_id(),
            task=task,
            evaluators=EVALUATORS,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
        )
        eval_columns = [c for c in df.columns if c.startswith("eval.")]
        print(df[["example_id", *eval_columns]].to_string() if not df.empty else "no rows")
    finally:
        instrumentation.flush_tracing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
