"""Create a labeled dataset, run the PDF pipeline as an Arize AX experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from config import (
    DEFAULT_DATASET_PREFIX,
    DEFAULT_EXPERIMENT_PREFIX,
    LAST_RUN_PATH,
    PDF_DIR,
    arize_project_name,
    require_arize,
)
from ingest import generate_pdfs, load_ground_truth, pdf_path_for
from instrumentation import ensure_tracing, flush_tracing


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def dataset_examples() -> list[dict[str, Any]]:
    fixture = load_ground_truth()
    examples = []
    for doc in fixture["documents"]:
        examples.append(
            {
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "notice_date": doc["notice_date"],
                "expected_retailer_name": doc["retailer_name"],
                "expected_deduction_amount": doc["deduction_amount_cents"],
                "expected_is_valid_claim": doc["is_valid_claim"],
                "is_trap": doc["is_trap"],
                "trap_type": doc.get("trap_type") or "",
                "notes": doc.get("notes") or "",
            }
        )
    return examples


def extract_task(dataset_row: dict[str, Any]) -> dict[str, Any]:
    """Experiment task: run the traced pipeline on one dataset example."""
    from extract import process_document

    filename = dataset_row.get("filename")
    doc_id = dataset_row.get("doc_id")
    if not filename:
        raise ValueError(f"dataset row missing filename: {dataset_row!r}")
    path = PDF_DIR / filename
    if not path.exists():
        generate_pdfs(overwrite=False)
    return process_document(
        path,
        doc_id=doc_id,
        extra_metadata={
            "is_trap": bool(dataset_row.get("is_trap")),
            "trap_type": dataset_row.get("trap_type") or "",
        },
    )


def run_experiment(
    *,
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    dry_run: bool = False,
    concurrency: int = 1,
) -> tuple[Any, Any]:
    from arize import ArizeClient

    from evaluators import EVALUATORS

    api_key, space_id, _project = require_arize()
    generate_pdfs(overwrite=False)
    examples = dataset_examples()

    client = ArizeClient(api_key=api_key)
    dataset_name = dataset_name or f"{DEFAULT_DATASET_PREFIX}-{_timestamp()}"
    experiment_name = experiment_name or f"{DEFAULT_EXPERIMENT_PREFIX}-{_timestamp()}"

    print(f"Creating dataset '{dataset_name}' with {len(examples)} examples...")
    dataset = client.datasets.create(
        space=space_id,
        name=dataset_name,
        examples=examples,
    )
    print(f"  dataset id: {dataset.id}")

    print(f"Running experiment '{experiment_name}' (concurrency={concurrency})...")
    experiment, experiment_df = client.experiments.run(
        name=experiment_name,
        dataset=dataset.id,
        space=space_id,
        task=extract_task,
        evaluators=EVALUATORS,
        concurrency=concurrency,
        dry_run=dry_run,
        set_global_tracer_provider=False,
    )
    _save_last_run(dataset, experiment, experiment_df, experiment_name, dataset_name)
    _print_summary(experiment_df)
    if experiment is not None:
        print(f"\nExperiment id: {experiment.id}")
        print(f"Dataset:       {dataset_name} ({dataset.id})")
        print(f"Project:       {arize_project_name()}")
        print("Open Arize AX → Datasets → this dataset → Experiments tab.")
    return experiment, experiment_df


def _save_last_run(
    dataset: Any,
    experiment: Any,
    experiment_df: Any,
    experiment_name: str,
    dataset_name: str,
) -> None:
    spans: list[dict[str, Any]] = []
    run_ids: list[str] = []

    if experiment_df is not None:
        records = experiment_df.to_dict(orient="records")
        for row in records:
            output = row.get("output") or row.get("task_output") or {}
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    output = {}
            if not isinstance(output, dict):
                output = {}
            spans.append(
                {
                    "doc_id": output.get("doc_id") or row.get("doc_id"),
                    "span_id": output.get("span_id"),
                    "trace_id": output.get("trace_id"),
                    "filename": output.get("filename"),
                    "parse_ok": output.get("parse_ok"),
                    "retailer_name": output.get("retailer_name"),
                    "deduction_amount": output.get("deduction_amount"),
                    "is_valid_claim": output.get("is_valid_claim"),
                }
            )
            for key in ("id", "run_id", "experiment_run_id"):
                if row.get(key):
                    run_ids.append(str(row[key]))
                    break

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "project_name": arize_project_name(),
        "dataset_id": getattr(dataset, "id", None),
        "dataset_name": dataset_name,
        "experiment_id": getattr(experiment, "id", None),
        "experiment_name": experiment_name,
        "run_ids": run_ids,
        "spans": spans,
    }
    LAST_RUN_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved run metadata to {LAST_RUN_PATH}")


def _print_summary(experiment_df: Any) -> None:
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

    fail_hints = []
    for col in experiment_df.columns:
        series = experiment_df[col]
        if "label" in col.lower() or col.endswith("label"):
            fails = [v for v in series.tolist() if str(v).lower() in {"fail", "incorrect", "0"}]
            if fails:
                fail_hints.append(f"{col}: {len(fails)} fail(s)")
    if fail_hints:
        print("\nVisible failures (expected from trap docs DN-009..DN-012):")
        for hint in fail_hints:
            print(f"  {hint}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the synthetic dataset and run the mixed-eval experiment."
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run locally without logging to Arize.")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    ensure_tracing()
    try:
        run_experiment(
            dataset_name=args.dataset_name,
            experiment_name=args.experiment_name,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
    finally:
        flush_tracing()


if __name__ == "__main__":
    main()
