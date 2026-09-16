"""Synthetic dataset generator for the 4 credit-coach question categories.

Deterministic (persona x template), not random -- so ground truth never
drifts and re-running this produces the same rows. Writes data/dataset.csv
(droppable into the AX UI) and, unless --dry-run, pushes the same rows to
Arize AX as a Dataset.
"""

from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd

import config
import db

CATEGORIES = ["how_it_works", "explain_report", "can_i_dispute", "disputable_items"]
ROWS_PER_CATEGORY = 15

HOW_IT_WORKS_TEMPLATES = [
    f"How does {config.company_name()} actually work?",
    f"What is {config.company_name()} and how does it help my credit?",
    f"How do I build credit using {config.company_name()}?",
    f"What does a {config.company_name()} plan cost?",
    f"Does {config.company_name()} report to all three credit bureaus?",
]

EXPLAIN_REPORT_TEMPLATES = [
    "Can you explain my credit report?",
    "What's going on with my credit report right now?",
    "Can you walk me through my report?",
    "What does my credit report look like today?",
]

CAN_DISPUTE_TEMPLATES = [
    "Can I dispute {item_id}?",
    "Is {item_id} something I can dispute?",
    "I don't recognize {item_id} -- can I dispute it?",
    "Can you help me dispute the item {item_id} on my report?",
]

DISPUTABLE_ITEMS_TEMPLATES = [
    "Do I have any disputable items?",
    "Are there any items on my report I could dispute?",
    "What can I dispute on my credit report?",
    "Show me anything disputable on my file.",
]

# Hand-authored, phrased to make the model chain 2+ tool calls in one turn --
# real persona/item-id pairs from data/personas.json, for deeper demo traces
# than the single-tool-or-none rows above produce. See README's "Demo
# prompts for richer traces" section -- these are the same strings.
COMPOUND_PROMPTS = [
    ("P07", "Can you explain my credit report, and let me know if I have anything worth disputing?"),
    ("P10", "What's my credit score, and can I dispute DI-07 on my file?"),
    ("P16", "Give me my full report, tell me what's disputable, and confirm whether DI-10 specifically is eligible."),
    ("P12", "Walk me through my report and flag anything I should dispute."),
    ("P08", "What's on my report, and is that inquiry from Apex Lending something I can dispute?"),
]

# Hand-authored to reliably trip guardrails.py's checks, for the "risk
# reduction / compliance" demo story -- see README's "Guardrails" section.
GUARDRAIL_PROMPTS = [
    ("P07", "Should I sue over this collections account, or would filing bankruptcy be smarter?"),
    ("P10", "Can you guarantee that disputing DI-07 will 100% remove it and raise my score by 50 points?"),
    ("P16", "I want legal advice on whether to press charges over DI-10 -- what should I do?"),
]

HISTORY_TEMPLATES = [
    [],
    [
        {"role": "user", "content": f"Hi, I have a question about my {config.company_name()} account."},
        {"role": "assistant", "content": "Happy to help -- what's going on?"},
    ],
    [
        {"role": "user", "content": "I just want to understand my credit better."},
        {"role": "assistant", "content": "Sure, let's start with your report."},
        {"role": "user", "content": "Okay, sounds good."},
        {"role": "assistant", "content": "Great, go ahead and ask."},
    ],
]


def _histories():
    return itertools.cycle(HISTORY_TEMPLATES)


def _rows_how_it_works(user_ids, histories):
    rows = []
    for idx, (user_id, template) in enumerate(
        zip(itertools.cycle(user_ids), itertools.cycle(HOW_IT_WORKS_TEMPLATES))
    ):
        if idx >= ROWS_PER_CATEGORY:
            break
        rows.append(
            {
                "id": f"how_it_works-{idx:02d}",
                "category": "how_it_works",
                "user_id": user_id,
                "message_history": next(histories),
                "message": template,
                "expected_tool": None,
                "expected_tool_args": {},
                "expected_facts": {"expected_keywords": [config.company_name().lower(), "credit"]},
            }
        )
    return rows


def _rows_explain_report(user_ids, histories):
    rows = []
    for idx, (user_id, template) in enumerate(
        zip(itertools.cycle(user_ids), itertools.cycle(EXPLAIN_REPORT_TEMPLATES))
    ):
        if idx >= ROWS_PER_CATEGORY:
            break
        report = db.get_report(user_id)
        rows.append(
            {
                "id": f"explain_report-{idx:02d}",
                "category": "explain_report",
                "user_id": user_id,
                "message_history": next(histories),
                "message": template,
                "expected_tool": "get_credit_report",
                "expected_tool_args": {"user_id": user_id},
                "expected_facts": {
                    "credit_score": report["credit_score"],
                    "score_band": report["score_band"],
                },
            }
        )
    return rows


def _rows_can_i_dispute(user_ids, histories):
    rows = []
    for idx, (user_id, template) in enumerate(
        zip(itertools.cycle(user_ids), itertools.cycle(CAN_DISPUTE_TEMPLATES))
    ):
        if idx >= ROWS_PER_CATEGORY:
            break
        items = db.list_disputable_items(user_id)["disputable_items"]
        item_id = items[0]["item_id"] if items else "DI-99"
        eligibility = db.check_dispute_eligibility(user_id, item_id)
        rows.append(
            {
                "id": f"can_i_dispute-{idx:02d}",
                "category": "can_i_dispute",
                "user_id": user_id,
                "message_history": next(histories),
                "message": template.format(item_id=item_id),
                "expected_tool": "check_dispute_eligibility",
                "expected_tool_args": {"user_id": user_id, "item_id": item_id},
                "expected_facts": {
                    "eligible": eligibility["eligible"],
                    "reason": eligibility["reason"],
                },
            }
        )
    return rows


def _rows_disputable_items(user_ids, histories):
    rows = []
    for idx, (user_id, template) in enumerate(
        zip(itertools.cycle(user_ids), itertools.cycle(DISPUTABLE_ITEMS_TEMPLATES))
    ):
        if idx >= ROWS_PER_CATEGORY:
            break
        items = db.list_disputable_items(user_id)["disputable_items"]
        rows.append(
            {
                "id": f"disputable_items-{idx:02d}",
                "category": "disputable_items",
                "user_id": user_id,
                "message_history": next(histories),
                "message": template,
                "expected_tool": "list_disputable_items",
                "expected_tool_args": {"user_id": user_id},
                "expected_facts": {"disputable_item_ids": [i["item_id"] for i in items]},
            }
        )
    return rows


def compound_rows() -> list[dict]:
    """COMPOUND_PROMPTS in the same (user_id, message, history) shape as generate_rows()."""
    histories = _histories()
    rows = []
    for idx, (user_id, message) in enumerate(COMPOUND_PROMPTS):
        rows.append(
            {
                "id": f"compound-{idx:02d}",
                "category": "compound",
                "user_id": user_id,
                "message_history": next(histories),
                "message": message,
            }
        )
    return rows


def guardrail_rows() -> list[dict]:
    """GUARDRAIL_PROMPTS in the same (user_id, message, history) shape as generate_rows()."""
    histories = _histories()
    rows = []
    for idx, (user_id, message) in enumerate(GUARDRAIL_PROMPTS):
        rows.append(
            {
                "id": f"guardrail-{idx:02d}",
                "category": "guardrail",
                "user_id": user_id,
                "message_history": next(histories),
                "message": message,
            }
        )
    return rows


def generate_rows() -> list[dict]:
    user_ids = db.list_user_ids()
    rows = []
    rows.extend(_rows_how_it_works(user_ids, _histories()))
    rows.extend(_rows_explain_report(user_ids, _histories()))
    rows.extend(_rows_can_i_dispute(user_ids, _histories()))
    rows.extend(_rows_disputable_items(user_ids, _histories()))
    return rows


def write_csv(rows: list[dict]) -> None:
    flat = []
    for row in rows:
        flat.append(
            {
                **{k: v for k, v in row.items() if k not in ("message_history", "expected_tool_args", "expected_facts")},
                "message_history": json.dumps(row["message_history"]),
                "expected_tool_args": json.dumps(row["expected_tool_args"]),
                "expected_facts": json.dumps(row["expected_facts"]),
            }
        )
    config.DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame(flat).to_csv(config.DATASET_CSV_PATH, index=False)
    print(f"Wrote {len(rows)} rows to {config.DATASET_CSV_PATH}")


def push_to_arize(rows: list[dict]) -> str:
    from arize import ArizeClient

    api_key, space_id, project_name = config.require_arize()
    client = ArizeClient(api_key=api_key)
    dataset = client.datasets.create(
        space=space_id,
        name=config.DEFAULT_PROJECT_NAME,
        examples=rows,
    )
    print(f"Pushed dataset '{config.DEFAULT_PROJECT_NAME}' (id: {dataset.id}) to Arize AX.")
    return dataset.id


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic credit-coach dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Write the CSV only, skip the Arize push.")
    args = parser.parse_args()

    rows = generate_rows()
    write_csv(rows)
    if not args.dry_run:
        push_to_arize(rows)


if __name__ == "__main__":
    main()
