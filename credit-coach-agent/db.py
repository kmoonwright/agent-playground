"""Fake local credit-profile store. No real users, no real company data."""

from __future__ import annotations

import json

from config import PERSONAS_PATH


def _load_personas() -> dict[str, dict]:
    raw = json.loads(PERSONAS_PATH.read_text())
    return {p["user_id"]: p for p in raw}


PERSONAS: dict[str, dict] = _load_personas()


def list_user_ids() -> list[str]:
    return list(PERSONAS.keys())


def get_persona(user_id: str) -> dict | None:
    return PERSONAS.get(user_id)


def get_report(user_id: str) -> dict:
    persona = get_persona(user_id)
    if persona is None:
        return {"error": f"no report on file for user_id '{user_id}'"}
    return {
        "user_id": persona["user_id"],
        "credit_score": persona["credit_score"],
        "score_band": persona["score_band"],
        "report_summary": persona["report_summary"],
        "accounts": persona["accounts"],
    }


def list_disputable_items(user_id: str) -> dict:
    persona = get_persona(user_id)
    if persona is None:
        return {"error": f"no report on file for user_id '{user_id}'"}
    return {"user_id": user_id, "disputable_items": persona["disputable_items"]}


def check_dispute_eligibility(user_id: str, item_id: str) -> dict:
    persona = get_persona(user_id)
    if persona is None:
        return {"error": f"no report on file for user_id '{user_id}'"}
    for item in persona["disputable_items"]:
        if item["item_id"] == item_id:
            return {
                "item_id": item_id,
                "eligible": item["dispute_eligible"],
                "reason": item["reason"],
            }
    return {
        "item_id": item_id,
        "eligible": False,
        "reason": "item not found on this report",
    }
