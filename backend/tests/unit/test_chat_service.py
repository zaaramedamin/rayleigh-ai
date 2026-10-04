import logging
from datetime import date

import pytest

from app.ai.llm.base import ChatMessage
from app.assistant.chat import (
    MAX_MESSAGE_CHARS,
    TEMPERATURE,
    build_messages,
    reply_to,
    select_history,
    system_prompt,
)
from tests.fakes import FakeLLM

TODAY = date(2026, 10, 4)


def _user(text: str) -> ChatMessage:
    return ChatMessage("user", text)


def _bot(text: str) -> ChatMessage:
    return ChatMessage("assistant", text)


def test_the_model_gets_the_earlier_turns_and_then_the_new_message() -> None:
    llm = FakeLLM("Paris.")
    history = [_user("Hello"), _bot("Hi. How can I help?")]

    reply = reply_to(llm, "  What is the capital of France?  ", history, today=TODAY)

    assert (reply.text, reply.truncated) == ("Paris.", False)
    system, messages, temperature = llm.chats[0]
    assert messages == [*history, _user("What is the capital of France?")]
    assert system == system_prompt(TODAY)
    assert temperature == TEMPERATURE


def test_the_notes_path_is_not_involved() -> None:
    llm = FakeLLM("Hello.")

    reply_to(llm, "Hi", today=TODAY)

    assert llm.calls == []  # `generate` is what answers from notes use
    assert llm.chats[0][1] == [_user("Hi")]


def test_the_instructions_say_the_notes_are_out_of_reach_and_give_the_date() -> None:
    prompt = system_prompt(TODAY)

    assert "cannot see the user's notes" in prompt
    assert '"My notes"' in prompt  # the name of the switch in the interface
    assert "Never invent what their notes contain" in prompt
    assert "no internet access" in prompt
    assert "Sunday 4 October 2026" in prompt
    assert "{" not in prompt


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_an_empty_message_is_refused_before_the_model_is_called(message: str) -> None:
    llm = FakeLLM()

    with pytest.raises(ValueError, match="empty"):
        reply_to(llm, message, today=TODAY)
    assert llm.chats == []


def test_a_message_that_is_too_long_is_refused() -> None:
    assert build_messages("x" * MAX_MESSAGE_CHARS)[-1].content == "x" * MAX_MESSAGE_CHARS

    with pytest.raises(ValueError, match=f"longer than {MAX_MESSAGE_CHARS}"):
        build_messages("x" * (MAX_MESSAGE_CHARS + 1))


def test_the_oldest_turns_are_dropped_when_the_conversation_is_too_long() -> None:
    history = [
        _user("first question"),
        _bot("a" * 60),
        _user("second question"),
        _bot("b" * 60),
        _user("third question"),
        _bot("c" * 60),
    ]

    kept = select_history(history, budget=160)

    assert kept == history[2:]


def test_what_is_kept_always_starts_with_something_the_user_said() -> None:
    history = [_user("q" * 50), _bot("a" * 50), _user("next"), _bot("fine")]

    # The first question no longer fits, so its answer goes with it.
    kept = select_history(history, budget=60)

    assert kept == [_user("next"), _bot("fine")]


def test_one_turn_bigger_than_the_whole_budget_leaves_no_history() -> None:
    assert select_history([_user("q"), _bot("a" * 500)], budget=100) == []
    assert build_messages("hello", [_user("q" * 9000)]) == [_user("hello")]


def test_blank_turns_are_ignored_and_the_rest_is_trimmed() -> None:
    history = [_user("  Hi  "), _bot("   "), _bot(" Hello. \n")]

    assert select_history(history) == [_user("Hi"), _bot("Hello.")]


def test_a_reply_that_was_cut_off_says_so() -> None:
    llm = FakeLLM("The first half of the")
    llm.truncated = True

    assert reply_to(llm, "Tell me a long story", today=TODAY).truncated is True


def test_what_was_said_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM("The reply mentions marmalade.")

    with caplog.at_level(logging.DEBUG):
        reply_to(llm, "A question about zeppelins", [_user("an earlier walrus")], today=TODAY)

    assert "chat finished" in caplog.text
    for word in ("marmalade", "zeppelins", "walrus"):
        assert word not in caplog.text
