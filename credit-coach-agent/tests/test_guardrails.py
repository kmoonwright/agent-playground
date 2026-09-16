"""Offline checks for guardrails.py. No network calls -- pure regex/logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import guardrails  # noqa: E402


def test_check_input_triggers_on_legal_advice_requests():
    result = guardrails.check_input("P01", "Should I sue over this, or file for bankruptcy?")
    assert result.triggered
    assert result.safe_response is not None


def test_check_input_does_not_trigger_on_benign_messages():
    result = guardrails.check_input("P01", "Can you explain my credit report?")
    assert not result.triggered
    assert result.safe_response is None


def test_check_output_triggers_on_guarantee_language():
    result = guardrails.check_output("I guarantee this will 100% remove the item.", [])
    assert result.triggered
    assert result.safe_response is not None


def test_check_output_triggers_on_score_mismatch_and_uses_real_score():
    tool_calls = [
        {
            "name": "get_credit_report",
            "args": {"user_id": "P01"},
            "result": json.dumps({"credit_score": 548, "score_band": "poor"}),
        }
    ]
    result = guardrails.check_output("Great news, your score is 750!", tool_calls)
    assert result.triggered
    assert "548" in result.safe_response


def test_check_output_does_not_trigger_on_compliant_consistent_reply():
    tool_calls = [
        {
            "name": "get_credit_report",
            "args": {"user_id": "P01"},
            "result": json.dumps({"credit_score": 548, "score_band": "poor"}),
        }
    ]
    result = guardrails.check_output("Your credit score is 548, which is in the poor band.", tool_calls)
    assert not result.triggered
    assert result.safe_response is None
