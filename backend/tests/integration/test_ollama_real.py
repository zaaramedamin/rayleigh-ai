"""Tests against the real local Ollama server. Skipped when it is not running."""

import pytest

from app.ai.llm.base import LLMError
from app.core.config import get_settings
from app.knowledge.components import create_llm

SETTINGS = get_settings()


def _why_unavailable() -> str | None:
    try:
        installed = create_llm(SETTINGS, timeout_seconds=3).list_models()
    except LLMError:
        return "Ollama is not running"
    wanted = SETTINGS.llm_model if ":" in SETTINGS.llm_model else f"{SETTINGS.llm_model}:latest"
    return None if wanted in installed else f"model {SETTINGS.llm_model} is not installed"


_REASON = _why_unavailable()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")


def test_the_configured_model_answers_a_test_prompt() -> None:
    reply = create_llm(SETTINGS).generate(
        "This is a connection test. Reply with the single word: ready", "Are you ready?"
    )

    assert "ready" in reply.lower()
