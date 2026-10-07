"""The TypeScript types the interface uses must keep matching the API the backend serves.

The frontend mirrors every response model by hand in `frontend/src/api/types.ts`. This test reads
that file and the live OpenAPI document, and fails when a field is added, renamed or made optional
on one side only, or when a new response model is not mirrored at all.
"""

import re
from pathlib import Path
from typing import Any

import pytest

from app.main import app

TYPES_FILE = Path(__file__).resolve().parents[3] / "frontend" / "src" / "api" / "types.ts"

# TypeScript interface -> the response model of the same shape.
MIRRORED = {
    "FileTypeInfo": "FileTypeInfo",
    "SearchResult": "SearchResult",
    "AskSource": "AskSource",
    "AskResponse": "AskResponse",
    "ChatAction": "ChatAction",
    "ChatResponse": "ChatResponse",
    "AssistantInfo": "IdentityOut",
    "Memory": "MemoryOut",
    "MemoryList": "MemoryList",
    "VoiceStatus": "VoiceStatus",
    "Transcription": "Transcription",
    "EmbeddingStatus": "EmbeddingStatus",
    "LibraryStatus": "LibraryStatus",
    "LLMStatus": "LLMStatus",
    "SystemStatus": "SystemStatus",
    "AuthStatus": "AuthStatus",
    "LibraryDocument": "DocumentOut",
    "DocumentList": "DocumentList",
    "LocationInfo": "LocationOut",
    "VersionInfo": "VersionOut",
    "DocumentDetail": "DocumentDetail",
    "ChunkInfo": "ChunkOut",
    "ChunkPage": "ChunkPage",
    "JobInfo": "JobOut",
    "LibraryFolder": "FolderOut",
    "FolderRemoval": "FolderRemoval",
    "SyncStatus": "SyncOut",
    "SettingsInfo": "SettingsOut",
    "Profile": "ProfileOut",
    "ConversationInfo": "ConversationOut",
    "ConversationList": "ConversationList",
    "StoredMessage": "MessageOut",
    "ConversationDetail": "ConversationDetail",
}

# Response models the interface reads in a simpler way than a named type: the client unwraps them.
UNWRAPPED = {
    "SearchResponse": "client.search returns the list inside",
    "FolderList": "client.folders returns the list inside",
    "TokenResponse": "client.login returns the token inside",
    "ClearedMemories": "client.clearMemories returns the count inside",
    "ClearedConversations": "client.deleteConversations returns the count inside",
    "FeedbackOut": "not used by the interface yet",
    "FeedbackList": "not used by the interface yet",
    "ClearedFeedback": "not used by the interface yet",
    "ProfileFieldInfo": "an inline element of Profile.fields",
    "ProfileValues": "Record<ProfileKey, string>",
    "HTTPValidationError": "error bodies are read as { detail }",
    "ValidationError": "error bodies are read as { detail }",
}

INTERFACE = re.compile(r"^export interface (\w+)(?: extends (\w+))? \{$")
FIELD = re.compile(r"^ {2}(\w+)(\??):")


def _parse_interfaces(source: str) -> dict[str, dict[str, bool]]:
    """Interface name -> {field: is_required}, own fields first and inherited ones merged in."""
    own: dict[str, dict[str, bool]] = {}
    parents: dict[str, str] = {}
    current: str | None = None
    for line in source.splitlines():
        header = INTERFACE.match(line)
        if header:
            current = header.group(1)
            own[current] = {}
            if header.group(2):
                parents[current] = header.group(2)
        elif line.startswith("}"):
            current = None
        elif current:
            field = FIELD.match(line)
            if field:
                own[current][field.group(1)] = field.group(2) != "?"

    def merged(name: str) -> dict[str, bool]:
        inherited = merged(parents[name]) if name in parents else {}
        return {**inherited, **own[name]}

    return {name: merged(name) for name in own}


def _schemas() -> dict[str, Any]:
    return app.openapi()["components"]["schemas"]


def _response_models() -> set[str]:
    """Every model a route answers with, directly or nested inside another response."""
    document = app.openapi()
    found: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                name = ref.rsplit("/", 1)[-1]
                if name not in found:
                    found.add(name)
                    visit(document["components"]["schemas"][name])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    for path in document["paths"].values():
        for operation in path.values():
            for status, response in operation.get("responses", {}).items():
                if status.startswith("2"):
                    visit(response)
    return found


@pytest.fixture(scope="module")
def interfaces() -> dict[str, dict[str, bool]]:
    if not TYPES_FILE.is_file():
        pytest.skip("the frontend is not in this checkout")
    return _parse_interfaces(TYPES_FILE.read_text(encoding="utf-8"))


def test_the_parser_reads_fields_and_inheritance(interfaces: dict[str, dict[str, bool]]) -> None:
    assert interfaces["AskSource"]["marker"] is True
    assert interfaces["AskSource"]["citation_id"] is True  # inherited from SearchResult
    assert interfaces["SearchResult"]["start_page"] is False  # optional


@pytest.mark.parametrize(("typescript", "model"), sorted(MIRRORED.items()))
def test_a_mirrored_type_has_the_fields_of_its_response_model(
    interfaces: dict[str, dict[str, bool]], typescript: str, model: str
) -> None:
    assert typescript in interfaces, f"types.ts has no interface {typescript}"
    schema = _schemas()[model]
    server_fields = set(schema["properties"])
    server_required = set(schema.get("required", []))
    client = interfaces[typescript]

    only_client = sorted(set(client) - server_fields)
    only_server = sorted(server_fields - set(client))
    assert set(client) == server_fields, (
        f"{typescript} and {model} differ: only in the interface {only_client}, "
        f"only on the server {only_server}"
    )
    optional_in_client = {name for name in server_required if not client[name]}
    assert not optional_in_client, (
        f"{model} always sends {sorted(optional_in_client)}, but {typescript} marks it optional"
    )


def test_every_response_model_is_mirrored_or_acknowledged() -> None:
    unknown = _response_models() - set(MIRRORED.values()) - set(UNWRAPPED)
    assert not unknown, (
        f"response models with no mirror in frontend/src/api/types.ts: {sorted(unknown)}. "
        "Mirror each one and add it to MIRRORED, or say in UNWRAPPED why the client reads it "
        "differently."
    )


def test_the_acknowledged_models_still_exist() -> None:
    assert set(UNWRAPPED) <= set(_schemas())
    assert set(MIRRORED.values()) <= set(_schemas())
