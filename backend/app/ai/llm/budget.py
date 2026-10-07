"""How much text fits in the model's context window, and what is cut first when it does not.

A model reads a limited number of tokens. Ollama does not refuse a prompt that is too long: it
silently drops the *start* of it, and the start is where the instructions are. So the application
keeps the prompt inside the window itself, and decides what to give up, in this order:

1. The instructions (system prompt) are never cut. Neither is the person's latest message: if the
   two together do not fit, the message is refused with a sentence that says so.
2. Notes given to the model for an answer are bounded by their own limit (see
   app.knowledge.answering.service), which always leaves room.
3. The earlier turns of a conversation take whatever room is left, newest first. The oldest turns
   are the first to go.

Tokens are estimated from characters, deliberately on the safe side.
"""

# What the model is asked to hold in mind (sent to Ollama as num_ctx), and the room kept free for
# its reply (sent as num_predict). The prompt may use the difference.
CONTEXT_TOKENS = 8192
REPLY_TOKENS = 1024
# With reasoning on, the reply budget is larger, and the prompt gets that much less.
THINKING_REPLY_TOKENS = 4096
# English averages about four characters per token. Other languages, numbers and code use more
# tokens for the same text, so three keeps a margin without wasting much room.
CHARS_PER_TOKEN = 3
# Message framing and the chat template cost a few tokens per turn.
FRAMING_CHARS = 200


def input_chars(context_tokens: int = CONTEXT_TOKENS, reply_tokens: int = REPLY_TOKENS) -> int:
    """The most characters a prompt may hold so that the reply still fits in the window."""
    return max(0, (context_tokens - reply_tokens) * CHARS_PER_TOKEN)


def history_room(limit: int, system_chars: int, message_chars: int) -> int:
    """Characters left for earlier turns once the instructions and the new message are placed.

    Raises ValueError when the instructions and the message alone do not fit: they are never cut.
    """
    room = limit - system_chars - message_chars - FRAMING_CHARS
    if room < 0:
        raise ValueError(
            "the message is too long for the model to read together with its instructions; "
            "shorten it or turn off what the assistant is told about you"
        )
    return room
