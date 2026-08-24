"""Shared paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
LAST_RUN_PATH = ROOT / ".last_run.json"

load_dotenv(ROOT / ".env")

KNOWN_RETAILERS = ("Walmart", "Target", "CVS", "Amazon")

# deduction_amount is integer US cents. Reject negatives and amounts above $100k.
AMOUNT_MIN_CENTS = 0
AMOUNT_MAX_CENTS = 10_000_000  # $100,000.00

DEFAULT_PROJECT_NAME = "pdf-extraction-demo"
DEFAULT_DATASET_PREFIX = "deduction-notices"
DEFAULT_EXPERIMENT_PREFIX = "pdf-extraction"
ANNOTATION_CONFIG_NAME = "extraction_quality"


def arize_space_id() -> str | None:
    return os.environ.get("ARIZE_SPACE_ID") or os.environ.get("ARIZE_SPACE")


def arize_api_key() -> str | None:
    return os.environ.get("ARIZE_API_KEY")


def arize_project_name() -> str:
    return os.environ.get("ARIZE_PROJECT_NAME") or DEFAULT_PROJECT_NAME


def openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def openai_model() -> str:
    return os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


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
            "https://app.arize.com (Settings → API Keys / Space Settings)."
        )
    return api_key, space_id, project_name
