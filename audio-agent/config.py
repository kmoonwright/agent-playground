"""Env/config. The only module that reads the environment directly."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_NAME = "audio-agent"

load_dotenv(ROOT / ".env", override=True)

# name -> (provider, model id). Both the analyze and respond graph nodes
# share this one table, so picking a different chat model is one flag,
# not two.
CHAT_MODELS = {
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "claude-haiku": ("anthropic", "claude-haiku-4-5"),
}
DEFAULT_CHAT_MODEL = "gpt-4o-mini"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def arize_api_key() -> str:
    return _env("ARIZE_API_KEY")


def arize_space_id() -> str:
    return _env("ARIZE_SPACE_ID")


def arize_project_name() -> str:
    return _env("ARIZE_PROJECT_NAME", DEFAULT_PROJECT_NAME)


def openai_api_key() -> str:
    return _env("OPENAI_API_KEY")


def anthropic_api_key() -> str:
    return _env("ANTHROPIC_API_KEY")


def whisper_model() -> str:
    return _env("AUDIO_AGENT_WHISPER_MODEL", "whisper-1")


def max_audio_attach_bytes() -> int:
    """Cap on raw audio bytes base64-encoded into a span attribute. Above
    this, the span records why it skipped attaching instead of bloating
    the span (or an OTLP export) with a huge data URI."""
    return int(_env("AUDIO_AGENT_MAX_ATTACH_BYTES", "2000000"))


def resolve_chat_model_name(cli_value: str | None = None) -> str:
    """CLI flag > env var > built-in default."""
    return cli_value or _env("AUDIO_AGENT_MODEL") or DEFAULT_CHAT_MODEL


def build_chat_model(name: str):
    """A langchain chat model instance, so LangChainInstrumentor auto-traces
    the analyze/respond nodes' LLM calls with zero manual span code."""
    if name not in CHAT_MODELS:
        raise SystemExit(f"Unknown model {name!r}. Choices: {', '.join(CHAT_MODELS)}")
    provider, model_id = CHAT_MODELS[name]
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_id, api_key=openai_api_key())
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model_id, api_key=anthropic_api_key())
