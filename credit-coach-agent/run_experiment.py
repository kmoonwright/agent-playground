"""Run the agent against every dataset row and log an Experiment with all 4 evals."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone

import agent
import config
import dataset_gen
import evaluators
import tracing


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def task(dataset_row: dict) -> dict:
    result = agent.run_agent(
        user_id=dataset_row["user_id"],
        message=dataset_row["message"],
        history=dataset_row["message_history"],
    )
    return dataclasses.asdict(result)


def run_experiment(*, dataset_name: str | None = None, experiment_name: str | None = None, concurrency: int = 1):
    from arize import ArizeClient

    api_key, space_id, project_name = config.require_arize()
    rows = dataset_gen.generate_rows()

    client = ArizeClient(api_key=api_key)
    dataset_name = dataset_name or f"{config.DEFAULT_PROJECT_NAME}-{_timestamp()}"
    experiment_name = experiment_name or f"credit-coach-experiment-{_timestamp()}"

    print(f"Creating dataset '{dataset_name}' with {len(rows)} rows...")
    dataset = client.datasets.create(space=space_id, name=dataset_name, examples=rows)
    print(f"  dataset id: {dataset.id}")

    print(f"Running experiment '{experiment_name}' (concurrency={concurrency})...")
    experiment, experiment_df = client.experiments.run(
        name=experiment_name,
        dataset=dataset.id,
        space=space_id,
        task=task,
        evaluators=evaluators.EVALUATORS,
        concurrency=concurrency,
        set_global_tracer_provider=False,
    )

    _print_summary(experiment_df)
    if experiment is not None:
        print(f"\nExperiment id: {experiment.id}")
        print(f"Dataset:       {dataset_name} ({dataset.id})")
        print(f"Project:       {project_name}")
        print("Open Arize AX -> Datasets -> this dataset -> Experiments tab.")
    return experiment, experiment_df


def _print_summary(experiment_df) -> None:
    if experiment_df is None or getattr(experiment_df, "empty", True):
        print("No experiment rows returned.")
        return
    print("\n=== experiment results ===")
    cols = [c for c in experiment_df.columns if "eval" in c.lower() or c in {"output", "example_id"}]
    show = cols or list(experiment_df.columns)
    try:
        print(experiment_df[show].to_string(index=False))
    except Exception:
        print(experiment_df.head().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent against the synthetic dataset as an Arize experiment.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    tracing.ensure_tracing()
    try:
        run_experiment(
            dataset_name=args.dataset_name,
            experiment_name=args.experiment_name,
            concurrency=args.concurrency,
        )
    finally:
        tracing.flush_tracing()


if __name__ == "__main__":
    main()
