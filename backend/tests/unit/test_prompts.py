"""The prompts are versioned, and none changes silently."""

import pytest

from app.ai.llm.base import ChatMessage
from app.knowledge.answering import rewrite, service
from app.knowledge.answering.prompts import ANSWER, PROMPTS, REWRITE, Prompt, versions
from tests.fakes import FakeLLM

# The exact text of every prompt, by version. To change a prompt: edit it, raise its version in
# prompts.py, and record the new fingerprint here (run `python -m app eval --answers` before and
# after, because the report prints these versions).
PINNED = {
    "answer": (1, "6e8917b995"),
    "rewrite": (1, "fb7c7adac9"),
    "summarize": (1, "52e3e2ad21"),
    "combine": (1, "b322afcb5b"),
    "compare": (1, "f2e9eb65a3"),
    "extract": (1, "883487bed8"),
    "agent": (1, "63a4427581"),
}


def test_every_prompt_is_pinned_at_its_version_and_its_exact_text() -> None:
    current = {name: (prompt.version, prompt.fingerprint) for name, prompt in PROMPTS.items()}

    assert current == PINNED, (
        "a prompt changed (or a new one was added). Raise its version in prompts.py, record "
        "the new fingerprint in PINNED, and compare the evaluation before and after."
    )


def test_a_prompt_changed_without_a_new_version_would_be_caught() -> None:
    edited = Prompt("answer", 1, ANSWER.system + " Be brief.")

    assert (edited.version, edited.fingerprint) != PINNED["answer"]


def test_the_fingerprint_follows_the_text_exactly() -> None:
    first, second = Prompt("x", 1, "Rules."), Prompt("x", 1, "Rules. ")

    assert first.fingerprint == Prompt("x", 1, "Rules.").fingerprint
    assert first.fingerprint != second.fingerprint and len(first.fingerprint) == 10


def test_a_prompt_is_registered_under_its_own_name_with_a_positive_version() -> None:
    for name, prompt in PROMPTS.items():
        assert prompt.name == name and prompt.version >= 1 and prompt.system.strip()


def test_the_report_labels_say_which_version_of_which_prompt() -> None:
    assert versions() == {
        name: f"v{version} ({fingerprint})" for name, (version, fingerprint) in PINNED.items()
    }
    assert ANSWER.label == f"answer v1 ({ANSWER.fingerprint})"


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_every_prompt_tells_the_model_that_what_it_reads_is_data_not_instructions(
    name: str,
) -> None:
    assert "data, not instructions" in PROMPTS[name].system


def test_the_modules_that_use_a_prompt_take_it_from_the_registry() -> None:
    assert service.SYSTEM_PROMPT == ANSWER.system
    system, _user = service.build_prompt("q", [], nonce="0" * 16)
    assert system == ANSWER.system
    llm = FakeLLM("a question?")
    rewrite.standalone_question(
        llm, "and its price?", [ChatMessage("user", "tell me about the car")]
    )
    assert llm.calls[0][0] == REWRITE.system == rewrite.SYSTEM_PROMPT
