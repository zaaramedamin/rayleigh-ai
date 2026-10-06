from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api import library_sync
from app.api.deps import get_session
from app.core.config import Settings, get_settings
from app.knowledge.profile.service import (
    PROFILE_FILENAME,
    clean_value,
    parse_profile,
    render_profile,
)
from app.knowledge.retrieval.service import retrieve
from app.main import app
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder

MODEL = "test/hashing-embedder"


@pytest.fixture
def wired(
    session: Session, make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[HashingEmbedder, QdrantVectorStore, Settings]]:
    settings = make_settings(embedding_model=MODEL)
    embedder = HashingEmbedder(model_name=MODEL)
    store = QdrantVectorStore.in_memory("profile-test", embedder.dimension)

    @contextmanager
    def locked(*_args: object, **_kwargs: object) -> Iterator[QdrantVectorStore]:
        yield store

    monkeypatch.setattr(library_sync, "locked_vector_store", locked)
    monkeypatch.setattr(library_sync, "load_embedder", lambda *_a, **_k: embedder)
    monkeypatch.setattr(library_sync, "is_model_downloaded", lambda *_a, **_k: True)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = lambda: session
    yield embedder, store, settings
    store.close()
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


FULL = {
    "name": "Sam",
    "location": "Berlin, Germany",
    "occupation": "I build a private assistant",
    "languages": "English and French",
    "interests": "climbing and cooking",
    "preferences": "short answers",
    "about": "",
}


# --- the note ---------------------------------------------------------------------------------


def test_the_profile_is_written_as_a_note_with_one_heading_per_field() -> None:
    text = render_profile(FULL)

    assert text.startswith("# About me\n")
    assert "## Where I live\n\nBerlin, Germany" in text
    assert "More about me" not in text  # empty fields are left out


def test_an_empty_profile_has_no_note() -> None:
    assert render_profile({}) == ""
    assert render_profile({"name": "   "}) == ""


def test_the_fields_survive_a_round_trip() -> None:
    parsed = parse_profile(render_profile(FULL))

    assert {k: v for k, v in parsed.items() if v} == {k: v for k, v in FULL.items() if v}


def test_a_value_cannot_fake_a_heading_or_carry_control_characters() -> None:
    cleaned = clean_value("first\n## My name\nforged\x00\x07 text")

    assert "##" not in cleaned
    assert "\x00" not in cleaned and "\x07" not in cleaned
    roundtrip = parse_profile(render_profile({"about": "line\n## My name\nforged"}))
    assert "name" not in roundtrip


def test_a_long_value_is_cut_to_the_limit() -> None:
    assert len(clean_value("x" * 10_000)) == 2000


# --- the API ----------------------------------------------------------------------------------


def test_a_new_profile_is_empty(client: TestClient, wired: tuple[object, ...]) -> None:
    body = client.get("/api/v1/profile").json()

    assert all(value == "" for value in body["values"].values())
    assert body["updated_at"] is None
    assert body["searchable"] is False
    assert [f["key"] for f in body["fields"]][0] == "name"


def test_saving_the_profile_makes_it_a_searchable_library_note(
    client: TestClient,
    wired: tuple[HashingEmbedder, QdrantVectorStore, Settings],
    session: Session,
) -> None:
    embedder, store, _settings = wired

    saved = client.put("/api/v1/profile", json=FULL)

    assert saved.status_code == 200
    assert saved.json()["values"]["location"] == "Berlin, Germany"
    assert saved.json()["searchable"] is True
    assert saved.json()["saved_as"] == PROFILE_FILENAME
    hits = retrieve(session, embedder, store, "where do I live", top_k=3)
    assert hits[0].source == PROFILE_FILENAME
    assert "Berlin" in hits[0].text
    assert hits[0].heading_path.endswith("Where I live")


def test_the_profile_shows_up_in_the_library_marked_as_such(
    client: TestClient, wired: tuple[object, ...]
) -> None:
    client.put("/api/v1/profile", json=FULL)

    documents = client.get("/api/v1/library/documents").json()["documents"]

    assert [(d["name"], d["is_profile"]) for d in documents] == [(PROFILE_FILENAME, True)]


def test_saving_again_replaces_the_note_instead_of_adding_another(
    client: TestClient,
    wired: tuple[HashingEmbedder, QdrantVectorStore, Settings],
    session: Session,
) -> None:
    embedder, store, _settings = wired
    client.put("/api/v1/profile", json=FULL)

    changed = client.put("/api/v1/profile", json={**FULL, "location": "Lisbon, Portugal"})

    assert changed.json()["values"]["location"] == "Lisbon, Portugal"
    documents = client.get("/api/v1/library/documents").json()["documents"]
    assert len(documents) == 1
    texts = " ".join(h.text for h in retrieve(session, embedder, store, "where do I live", top_k=5))
    assert "Lisbon" in texts
    assert "Berlin" not in texts


def test_clearing_a_field_everywhere_removes_the_note(
    client: TestClient, wired: tuple[object, ...]
) -> None:
    client.put("/api/v1/profile", json=FULL)

    client.put("/api/v1/profile", json={})

    assert client.get("/api/v1/library/documents").json()["documents"] == []
    assert client.get("/api/v1/profile").json()["updated_at"] is None


def test_the_profile_can_be_deleted(client: TestClient, wired: tuple[object, ...]) -> None:
    client.put("/api/v1/profile", json=FULL)

    assert client.delete("/api/v1/profile").status_code == 204

    assert client.get("/api/v1/library/documents").json()["documents"] == []


def test_the_profile_is_kept_even_if_the_search_index_is_unavailable(
    client: TestClient, wired: tuple[object, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(library_sync, "is_model_downloaded", lambda *_a, **_k: False)

    saved = client.put("/api/v1/profile", json=FULL)

    assert saved.status_code == 200
    assert saved.json()["searchable"] is False
    assert saved.json()["values"]["name"] == "Sam"


def test_unknown_or_oversized_fields_are_refused(
    client: TestClient, wired: tuple[object, ...]
) -> None:
    too_long = client.put("/api/v1/profile", json={"name": "x" * 2001})

    assert too_long.status_code == 422


def test_the_profile_text_stays_out_of_the_log(
    client: TestClient, wired: tuple[object, ...], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")

    client.put("/api/v1/profile", json=FULL)

    assert "Berlin" not in caplog.text
    assert "Sam" not in caplog.text


def test_the_stored_copy_is_not_written_to_the_notes_folder(
    client: TestClient, wired: tuple[object, ...], tmp_path: Path
) -> None:
    client.put("/api/v1/profile", json=FULL)

    assert not list(tmp_path.glob("*.md"))
