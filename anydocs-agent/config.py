"""Env/config. The only module that reads the environment directly."""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "docs"
EVAL_DATASET_PATH = ROOT / "data" / "eval_dataset.json"
EMBEDDING_CACHE_PATH = ROOT / "cache" / "embeddings.json"
DEFAULT_PROJECT_NAME = "anydocs-agent"

load_dotenv(ROOT / ".env", override=True)


def _env(name: str, default: str = "") -> str:
    import os

    return os.environ.get(name, default)


def arize_api_key() -> str:
    return _env("ARIZE_API_KEY")


def arize_space_id() -> str:
    return _env("ARIZE_SPACE_ID")


def arize_project_name() -> str:
    return _env("ARIZE_PROJECT_NAME", DEFAULT_PROJECT_NAME)


def openai_api_key() -> str:
    return _env("OPENAI_API_KEY")


def openai_model() -> str:
    return _env("OPENAI_MODEL", "gpt-4o-mini")


def require_arize() -> None:
    if not arize_api_key() or not arize_space_id():
        raise SystemExit(
            "ARIZE_API_KEY and ARIZE_SPACE_ID are required for this — "
            "datasets/experiments are server-side Arize AX resources, "
            "unlike tracing there's no local fallback. Copy .env.example to "
            ".env and fill them in from https://app.arize.com."
        )
