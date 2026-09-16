"""Shared paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PERSONAS_PATH = DATA_DIR / "personas.json"
DATASET_CSV_PATH = DATA_DIR / "dataset.csv"

load_dotenv(ROOT / ".env")

MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 6
DEFAULT_PROJECT_NAME = "credit-coach-agent"


def openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def company_name() -> str:
    return os.environ.get("COMPANY_NAME") or "Acme"


def agent_name() -> str:
    return os.environ.get("AGENT_NAME") or "Sage"


def arize_space_id() -> str | None:
    return os.environ.get("ARIZE_SPACE_ID") or os.environ.get("ARIZE_SPACE")


def arize_api_key() -> str | None:
    return os.environ.get("ARIZE_API_KEY")


def arize_project_name() -> str:
    return os.environ.get("ARIZE_PROJECT_NAME") or DEFAULT_PROJECT_NAME


def require_openai() -> str:
    api_key = openai_api_key()
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and fill in a key."
        )
    return api_key


def require_arize() -> tuple[str, str, str]:
    api_key = arize_api_key()
    space_id = arize_space_id()
    project_name = arize_project_name()
    missing = [
        name
        for name, value in (
            ("ARIZE_API_KEY", api_key),
            ("ARIZE_SPACE_ID", space_id),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required env vars: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in values from "
            "https://app.arize.com (Space Settings)."
        )
    return api_key, space_id, project_name
