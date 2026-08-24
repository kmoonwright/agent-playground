"""Shared paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRS_DIR = DATA_DIR / "prs"
OUT_DIR = ROOT / "out"

load_dotenv(ROOT / ".env")

DEFAULT_PROJECT_NAME = "pr-review-agent"
DEFAULT_MAX_ITERATIONS = 8

# Patch budget. Truncated patches are marked so the agent reads the file
# instead of treating it as missing. Same caps as the OpenHands workshop.
MAX_TOTAL_PATCH_CHARS = 60_000
MAX_FILE_PATCH_CHARS = 8_000

MAX_ATTR_CHARS = 8_000


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


def github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or None


def require_openai() -> str:
    key = openai_api_key()
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key."
        )
    return key
