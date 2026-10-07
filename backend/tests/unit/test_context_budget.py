"""The prompt always fits the model's window, and the instructions are the last thing to be cut."""

from datetime import date

import pytest

import app.assistant.chat as chat_module
from app.ai.llm.base import ChatMessage, ToolSpec
from app.ai.llm.budget import (
    CHARS_PER_TOKEN,
    CONTEXT_TOKENS,
    FRAMING_CHARS,
    REPLY_TOKENS,
    THINKING_REPLY_TOKENS,
    history_room,
    input_chars,
)
from app.ai.llm.ollama import DEFAULT_NUM_CTX, OllamaProvider
from app.assistant.chat import (
    MAX_HISTORY_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_PROFILE_FIELD_CHARS,
    AssistantContext,
    reply_to,
    system_prompt,
)
from app.assistant.identity import MAX_ROLE_CHARS, Identity
from app.assistant.memory import MAX_MEMORY_CHARS
from app.knowledge.answering.rewrite import (
    MAX_TURN_CHARS,
    MAX_TURNS_USED,
    standalone_question,
)
from app.knowledge.answering.rewrite import (
    SYSTEM_PROMPT as REWRITE_SYSTEM,
)
from app.knowledge.answering.service import MAX_CONTEXT_CHARS, build_prompt, select_context
from app.knowledge.retrieval.service import MAX_QUERY_CHARS, RetrievedChunk
from tests.fakes import FakeLLM

TODAY = date(2026, 10, 7)
WINDOW = input_chars()


def turns(count: int, size: int = 400) -> list[ChatMessage]:
    """A long conversation, oldest first, each turn numbered so the order can be checked."""
    out: list[ChatMessage] = []
    for number in range(count):
        out.append(ChatMessage("user", f"q{number:04d} " + "x" * size))
        out.append(ChatMessage("assistant", f"a{number:04d} " + "y" * size))
    return out


def sent_size(system: str, messages: list[ChatMessage]) -> int:
    return len(system) + sum(len(m.content) for m in messages) + FRAMING_CHARS


# --- the numbers -----------------------------------------------------------------------------


def test_the_provider_asks_ollama_for_the_window_the_budget_assumes() -> None:
    quiet = OllamaProvider("http://127.0.0.1:11434", "m")
    thinking = OllamaProvider("http://127.0.0.1:11434", "m", think=True)

    body = quiet._chat_body("s", [ChatMessage("user", "u")], temperature=0)
    thinking_body = thinking._chat_body("s", [ChatMessage("user", "u")], temperature=0)

    assert DEFAULT_NUM_CTX == CONTEXT_TOKENS
    assert body["options"]["num_ctx"] == CONTEXT_TOKENS
    assert body["options"]["num_predict"] == REPLY_TOKENS == quiet.reply_tokens
    assert thinking_body["options"]["num_predict"] == THINKING_REPLY_TOKENS


def test_what_a_prompt_may_hold_leaves_room_for_the_reply() -> None:
    assert input_chars(8192, 1024) == (8192 - 1024) * CHARS_PER_TOKEN
    assert input_chars(1000, 2000) == 0  # a reply budget larger than the window leaves nothing
    quiet = OllamaProvider("http://127.0.0.1:11434", "m")
    thinking = OllamaProvider("http://127.0.0.1:11434", "m", think=True)
    # Reasoning reserves more room for the reply, so the prompt may be shorter.
    assert quiet.input_chars - thinking.input_chars == (
        (THINKING_REPLY_TOKENS - REPLY_TOKENS) * CHARS_PER_TOKEN
    )


def test_the_room_for_history_is_what_the_instructions_and_the_message_leave() -> None:
    assert history_room(10_000, 3_000, 2_000) == 10_000 - 3_000 - 2_000 - FRAMING_CHARS


def test_instructions_and_message_that_do_not_fit_together_are_refused_not_cut() -> None:
    with pytest.raises(ValueError, match="too long for the model to read together"):
        history_room(5_000, 3_000, 2_000)


# --- a long conversation keeps working -------------------------------------------------------


def test_a_very_long_conversation_still_fits_with_the_newest_turns_and_the_full_instructions() -> (
    None
):
    llm = FakeLLM("ok")
    history = turns(300)  # about 240,000 characters

    reply_to(llm, "What now?", history, today=TODAY)

    system, messages, _temperature = llm.chats[0]
    assert system == system_prompt(TODAY)  # whole: nothing was taken from the instructions
    assert sent_size(system, messages) <= WINDOW
    assert messages[-1] == ChatMessage("user", "What now?")
    assert messages[0].role == "user"  # never starts on a reply whose question was dropped
    kept = [m.content[:5] for m in messages[:-1]]
    assert kept[-1] == "a0299" and "q0000" not in kept  # the newest are kept, the oldest are gone
    assert kept == sorted(kept, key=lambda c: (c[1:], c[0] != "q"))  # in order, no gaps in between


def test_history_is_cut_before_anything_else_when_the_instructions_are_at_their_largest() -> None:
    # The worst case the settings allow: the longest role, a full profile, a full memory.
    identity = Identity(role="r" * MAX_ROLE_CHARS)
    profile = [(f"Field {n}", "p" * MAX_PROFILE_FIELD_CHARS) for n in range(4)]
    memories = ["m" * MAX_MEMORY_CHARS for _ in range(5)]
    context = AssistantContext(identity=identity, profile=profile, memories=memories)
    llm = FakeLLM("ok")
    message = "z" * MAX_MESSAGE_CHARS  # and the longest message

    reply_to(llm, message, turns(100), today=TODAY, context=context)

    system, messages, _temperature = llm.chats[0]
    assert (
        "r" * MAX_ROLE_CHARS in system
        and "=== PROFILE END" in system
        and "=== MEMORY END" in system
    )
    assert messages[-1].content == message  # the message is whole as well
    assert sent_size(system, messages) <= WINDOW
    # Before this budget the same request needed more than the window: the worst case overflowed.
    assert len(system) + MAX_MESSAGE_CHARS + MAX_HISTORY_CHARS + FRAMING_CHARS > WINDOW


def test_the_tools_the_model_is_offered_take_room_from_the_history_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = AssistantContext(can_act=True)
    instructions = len(system_prompt(TODAY, context, can_remember=False, nonce="0" * 16))
    limit = instructions + 3_000
    history = turns(20, size=200)
    big_tools = [ToolSpec(f"tool_{n}", "d" * 500, {"type": "object"}) for n in range(5)]

    def kept(tools: list[ToolSpec]) -> int:
        monkeypatch.setattr(chat_module, "tool_specs", lambda **_kwargs: tools)
        llm = FakeLLM("ok")
        llm.input_chars = limit  # type: ignore[attr-defined]
        reply_to(llm, "Hi", history, today=TODAY, context=context)
        return len(llm.chats[0][1])

    assert kept(big_tools) < kept([])


def test_a_message_too_long_to_share_the_window_with_its_instructions_is_refused() -> None:
    llm = FakeLLM("never used")
    llm.input_chars = len(system_prompt(TODAY)) + 100  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="too long for the model to read together"):
        reply_to(llm, "word " * 200, today=TODAY, context=AssistantContext(can_act=False))

    assert llm.chats == []  # the model was not called with a damaged prompt


def test_a_model_that_reserves_more_room_for_its_reply_gets_a_shorter_history() -> None:
    # With long instructions the room, not the 8,000-character history cap, is what limits history.
    context = AssistantContext(
        identity=Identity(role="r" * MAX_ROLE_CHARS),
        profile=[(f"Field {n}", "p" * MAX_PROFILE_FIELD_CHARS) for n in range(4)],
        memories=["m" * MAX_MEMORY_CHARS for _ in range(5)],
    )
    history = turns(60)

    def kept(limit: int) -> int:
        llm = FakeLLM("ok")
        llm.input_chars = limit  # type: ignore[attr-defined]
        reply_to(llm, "Hi", history, today=TODAY, context=context)
        return len(llm.chats[0][1])

    quiet = OllamaProvider("http://127.0.0.1:11434", "m").input_chars
    thinking = OllamaProvider("http://127.0.0.1:11434", "m", think=True).input_chars
    assert thinking < quiet and kept(thinking) < kept(quiet)


# --- the other prompts are bounded by their own limits, which always fit ---------------------


def _note(number: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        citation_id=f"{number}:0",
        document_id=number,
        chunk_index=0,
        score=0.9,
        source=f"file-{number}.md",
        heading_path="A > B",
        start_line=1,
        end_line=2,
        text=text,
    )


def test_the_prompt_for_answering_from_notes_fits_even_with_the_longest_question() -> None:
    notes = [_note(n, "n" * 900) for n in range(40)]  # far more than may be used
    chosen = select_context(notes, min_score=0.3)

    system, user = build_prompt("q" * MAX_QUERY_CHARS, chosen, nonce="0123456789abcdef")

    assert sum(len(n.text) for n in chosen) <= MAX_CONTEXT_CHARS + 900
    assert len(system) + len(user) + FRAMING_CHARS <= WINDOW


def test_the_prompt_for_rewriting_a_follow_up_fits_with_the_longest_history() -> None:
    llm = FakeLLM("a question?")
    history = [ChatMessage("user" if n % 2 == 0 else "assistant", "w" * 5000) for n in range(40)]

    standalone_question(llm, "q" * MAX_QUERY_CHARS, history)

    system, user = llm.calls[0]
    assert system == REWRITE_SYSTEM
    # Only the last few turns are shown, each cut short.
    assert user.count("w" * MAX_TURN_CHARS) == MAX_TURNS_USED
    assert len(system) + len(user) + FRAMING_CHARS <= WINDOW
