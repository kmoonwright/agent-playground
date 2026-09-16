"""Offline checks for dataset_gen's COMPOUND_PROMPTS and generate_traces' pool logic.

No network calls -- pure data/pool-building checks.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset_gen  # noqa: E402
import db  # noqa: E402
import generate_traces  # noqa: E402

ITEM_ID_RE = re.compile(r"\bDI-\d+\b")


def _assert_prompts_reference_real_personas_and_items(prompts):
    for user_id, message in prompts:
        persona = db.get_persona(user_id)
        assert persona is not None, f"{user_id} not in personas.json"
        item_ids_on_file = {item["item_id"] for item in persona["disputable_items"]}
        for item_id in ITEM_ID_RE.findall(message):
            assert item_id in item_ids_on_file, (
                f"{message!r} references {item_id}, not on {user_id}'s disputable_items"
            )


def test_compound_prompts_reference_real_personas_and_items():
    _assert_prompts_reference_real_personas_and_items(dataset_gen.COMPOUND_PROMPTS)


def test_guardrail_prompts_reference_real_personas_and_items():
    _assert_prompts_reference_real_personas_and_items(dataset_gen.GUARDRAIL_PROMPTS)


def test_compound_rows_have_the_shape_run_agent_needs():
    rows = dataset_gen.compound_rows()
    assert len(rows) == len(dataset_gen.COMPOUND_PROMPTS)
    for row in rows:
        assert row["user_id"]
        assert row["message"]
        assert isinstance(row["message_history"], list)


def test_guardrail_rows_have_the_shape_run_agent_needs():
    rows = dataset_gen.guardrail_rows()
    assert len(rows) == len(dataset_gen.GUARDRAIL_PROMPTS)
    for row in rows:
        assert row["user_id"]
        assert row["message"]
        assert isinstance(row["message_history"], list)


def test_build_pool_and_select_are_pure_and_respect_count():
    pool = generate_traces.build_pool(compound_only=True)
    assert pool == dataset_gen.compound_rows()

    guardrail_pool = generate_traces.build_pool(guardrail_only=True)
    assert guardrail_pool == dataset_gen.guardrail_rows()

    selected = generate_traces.select(pool, 12)
    assert len(selected) == 12

    full_pool = generate_traces.build_pool()
    assert len(full_pool) == (
        len(dataset_gen.generate_rows()) + len(dataset_gen.COMPOUND_PROMPTS) + len(dataset_gen.GUARDRAIL_PROMPTS)
    )
