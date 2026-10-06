from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ai.llm.base import LLMError, LLMTimeoutError, LLMUnavailableError
from app.api.deps import get_session
from app.api.v1 import system as system_module
from app.core.config import Settings, get_settings
from app.knowledge.chunking.service import chunk_document
from app.main import app
from app.storage.files import save_file
from tests.fakes import FakeLLM


@pytest.fixture
def client(
    session: Session, make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(system_module, "create_llm", lambda *_a, **_k: FakeLLM(installed=["m:1b"]))
    app.dependency_overrides[get_settings] = lambda: make_settings(llm_model="m:1b")
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


def _llm_fails(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(
        system_module, "create_llm", lambda *_a, **_k: FakeLLM(error=error, installed=[])
    )


def test_an_empty_library_reports_zero_counts(client: TestClient) -> None:
    body = client.get("/api/v1/system/status").json()

    assert body["library"] == {
        "documents": 0,
        "chunks": 0,
        "searchable_documents": 0,
        "pending_documents": 0,
        "missing_documents": 0,
    }
    assert body["embedding"]["downloaded"] is False


def test_documents_waiting_for_indexing_are_counted_as_pending(
    client: TestClient, session: Session, data_dir: Path
) -> None:
    document = save_file(session, data_dir, b"# Oats\n\nOats are high in fibre.\n", "oats.md")
    chunk_document(session, data_dir, document, 1000, 50)

    library = client.get("/api/v1/system/status").json()["library"]

    assert library["documents"] == 1
    assert library["chunks"] >= 1
    assert (library["searchable_documents"], library["pending_documents"]) == (0, 1)


def test_an_installed_model_is_ready(client: TestClient) -> None:
    llm = client.get("/api/v1/system/status").json()["llm"]

    assert llm == {"model": "m:1b", "state": "ready", "hint": None}


def test_a_missing_model_says_how_to_install_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(system_module, "create_llm", lambda *_a, **_k: FakeLLM(installed=["x:1"]))

    llm = client.get("/api/v1/system/status").json()["llm"]

    assert llm["state"] == "model_missing"
    assert "ollama pull m:1b" in llm["hint"]


def test_ollama_not_running_is_reported_not_raised(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_fails(monkeypatch, LLMUnavailableError("connection refused at http://127.0.0.1:11434"))

    response = client.get("/api/v1/system/status")

    assert response.status_code == 200
    assert response.json()["llm"]["state"] == "not_running"


def test_a_model_server_that_does_not_answer_in_time_counts_as_not_running(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_fails(monkeypatch, LLMTimeoutError("no answer"))

    assert client.get("/api/v1/system/status").json()["llm"]["state"] == "not_running"


def test_error_detail_from_the_model_server_is_not_relayed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _llm_fails(monkeypatch, LLMError("secret detail that should stay in the server"))

    response = client.get("/api/v1/system/status")

    assert response.json()["llm"]["state"] == "error"
    assert "secret detail" not in response.text


def test_the_status_never_contains_note_text_or_file_names(
    client: TestClient, session: Session, data_dir: Path
) -> None:
    save_file(session, data_dir, b"my private diary entry", "diary.md")

    text = client.get("/api/v1/system/status").text

    assert "private diary" not in text
    assert "diary.md" not in text


# --- the settings, read-only ---------------------------------------------------------------


def test_the_settings_are_readable_and_hold_no_paths_or_secrets(
    client: TestClient, data_dir: Path
) -> None:
    body = client.get("/api/v1/system/settings").json()

    assert body["search_mode"] in {"vector", "keyword", "hybrid"}
    assert body["embedding_model"] and body["llm_model"] and body["version"]
    assert body["library_encrypted"] is False and isinstance(body["access_required"], bool)
    text = str(body).lower()
    for forbidden in (str(data_dir).lower(), "passphrase", "password", "token", "secret"):
        assert forbidden not in text, forbidden
    assert set(body) >= {"chunk_size_chars", "pdf_max_pages", "parser_timeout_seconds"}


def test_the_settings_cannot_be_changed_through_the_api(client: TestClient) -> None:
    for method in ("put", "post", "patch", "delete"):
        assert getattr(client, method)("/api/v1/system/settings").status_code == 405, method
