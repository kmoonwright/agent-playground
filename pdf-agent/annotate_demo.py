"""Demonstrate SDK-based annotation of a span (and optionally an experiment run).

This is the answer to "can a specific run be annotated via an API from an
external system?" — yes. This script is that external system.

Prerequisite (one-time, in Arize AX or via ax CLI): an annotation config
named `extraction_quality` must exist in the space. See README.md.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from config import ANNOTATION_CONFIG_NAME, LAST_RUN_PATH, require_arize


def load_last_run() -> dict[str, Any]:
    if not LAST_RUN_PATH.exists():
        raise SystemExit(
            f"No {LAST_RUN_PATH.name} found. Run extract.py or run_experiment.py first."
        )
    return json.loads(LAST_RUN_PATH.read_text())


def pick_span(last_run: dict[str, Any], span_id: str | None, doc_id: str | None) -> dict[str, Any]:
    spans = [s for s in last_run.get("spans") or [] if s.get("span_id")]
    if span_id:
        for span in spans:
            if span["span_id"] == span_id:
                return span
        return {"span_id": span_id, "doc_id": doc_id}
    if doc_id:
        for span in spans:
            if span.get("doc_id") == doc_id:
                return span
        raise SystemExit(f"No saved span for doc_id={doc_id}")
    # Prefer a trap document so the live demo annotates a visible failure.
    for span in spans:
        if str(span.get("doc_id") or "").startswith("DN-011"):
            return span
    if spans:
        return spans[-1]
    raise SystemExit("No span ids saved. Re-run the pipeline so span ids are recorded.")


def annotate_span(
    *,
    span_id: str,
    label: str,
    score: float | None,
    updated_by: str,
    notes: str | None,
) -> Any:
    from arize import ArizeClient

    api_key, space_id, project_name = require_arize()
    client = ArizeClient(api_key=api_key)

    row: dict[str, Any] = {
        "context.span_id": span_id,
        f"annotation.{ANNOTATION_CONFIG_NAME}.label": label,
        f"annotation.{ANNOTATION_CONFIG_NAME}.updated_by": updated_by,
    }
    if score is not None:
        row[f"annotation.{ANNOTATION_CONFIG_NAME}.score"] = score
    if notes:
        row["annotation.notes"] = notes

    df = pd.DataFrame([row])
    print(f"Pushing annotation to project '{project_name}' via client.spans.update_annotations:")
    print(df.to_string(index=False))
    response = client.spans.update_annotations(
        space_id=space_id,
        project_name=project_name,
        dataframe=df,
        validate=True,
    )
    return response


def annotate_experiment_run(
    *,
    experiment_id: str,
    dataset: str | None,
    run_id: str,
    label: str,
    score: float | None,
    notes: str | None,
) -> None:
    from arize import ArizeClient

    api_key, space_id, _project = require_arize()
    client = ArizeClient(api_key=api_key)

    try:
        from arize.experiments.types import AnnotateRecordInput, AnnotationInput
    except ImportError:
        from arize._generated.models import AnnotateRecordInput, AnnotationInput  # type: ignore

    values = [AnnotationInput(name=ANNOTATION_CONFIG_NAME, label=label, score=score)]
    if notes:
        values.append(AnnotationInput(name="reviewer_notes", text=notes))

    print(
        f"Pushing experiment-run annotation via client.experiments.annotate_runs "
        f"(experiment={experiment_id}, run={run_id})..."
    )
    client.experiments.annotate_runs(
        experiment=experiment_id,
        dataset=dataset,
        space=space_id,
        annotations=[
            AnnotateRecordInput(record_id=run_id, values=values),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a span (and optionally an experiment run) via the Arize SDK."
    )
    parser.add_argument("--span-id", default=None, help="Span id to annotate. Defaults to last saved trap span.")
    parser.add_argument("--doc-id", default=None, help="Pick a saved span by document id (e.g. DN-011).")
    parser.add_argument(
        "--label",
        default="incorrect",
        help=f"Label for the '{ANNOTATION_CONFIG_NAME}' annotation config.",
    )
    parser.add_argument("--score", type=float, default=0.0)
    parser.add_argument("--updated-by", default="pdf-demo-external-system")
    parser.add_argument(
        "--notes",
        default="Annotated via annotate_demo.py (SDK) — no UI click.",
    )
    parser.add_argument(
        "--also-experiment-run",
        action="store_true",
        help="Also annotate the first saved experiment run id via experiments.annotate_runs.",
    )
    parser.add_argument("--run-id", default=None, help="Experiment run id (defaults to first saved run).")
    args = parser.parse_args()

    last_run = load_last_run()
    span = pick_span(last_run, args.span_id, args.doc_id)
    span_id = span["span_id"]
    print(
        f"Annotating span {span_id} "
        f"(doc_id={span.get('doc_id')}, trace_id={span.get('trace_id')})"
    )

    response = annotate_span(
        span_id=span_id,
        label=args.label,
        score=args.score,
        updated_by=args.updated_by,
        notes=args.notes,
    )
    print("spans.update_annotations response:", response)
    print(
        "Refresh the trace in Arize AX — the annotation should appear on the span "
        f"under '{ANNOTATION_CONFIG_NAME}' with no UI interaction."
    )

    if args.also_experiment_run:
        experiment_id = last_run.get("experiment_id")
        run_id = args.run_id or (last_run.get("run_ids") or [None])[0]
        if not experiment_id or not run_id:
            raise SystemExit(
                "No experiment_id / run_id in .last_run.json. "
                "Run run_experiment.py first, or pass --run-id."
            )
        annotate_experiment_run(
            experiment_id=experiment_id,
            dataset=last_run.get("dataset_id"),
            run_id=run_id,
            label=args.label,
            score=args.score,
            notes=args.notes,
        )
        print("experiments.annotate_runs completed.")


if __name__ == "__main__":
    main()
