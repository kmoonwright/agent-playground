"""Offline checks for generate_sessions.py's pure session-plan logic.

No network calls -- session_messages()/build_session_plans() are pure data
builders; run_session() (which actually calls the model) is not exercised
here, only its inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
import generate_sessions  # noqa: E402


def test_session_messages_are_four_ordered_turns_referencing_real_data():
    for user_id in db.list_user_ids():
        messages = generate_sessions.session_messages(user_id)
        assert len(messages) == 4
        assert all(isinstance(m, str) and m for m in messages)
        # the 4th message is the dispute-specific one -- if the persona has a
        # real disputable item, its id must actually appear in the message
        items = db.list_disputable_items(user_id)["disputable_items"]
        if items:
            assert items[0]["item_id"] in messages[3]


def test_build_session_plans_cycles_personas_and_respects_count():
    plans = generate_sessions.build_session_plans(5)
    assert len(plans) == 5
    session_ids = {p["session_id"] for p in plans}
    assert len(session_ids) == 5  # all unique
    for plan in plans:
        assert plan["user_id"] in db.list_user_ids()
        assert len(plan["messages"]) == 4

    # more sessions than personas -> cycles, but session_ids stay unique
    many = generate_sessions.build_session_plans(len(db.list_user_ids()) + 3)
    assert len({p["session_id"] for p in many}) == len(many)
