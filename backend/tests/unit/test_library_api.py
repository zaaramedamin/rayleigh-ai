import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api import library_sync
from app.api.deps import get_session
from app.core.config import Settings, get_settings
from app.knowledge.library.state import LIBRARY_FILENAME
from app.main import app
from app.storage.models import Document
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder

MODEL = "test/hashing-embedder"


class Library:
    """A notes folder, an in-memory search index and the API wired to them."""

    def __init__(self, tmp_path: Path, settings: Settings, store: QdrantVectorStore) -> None:
        self.notes = tmp_path / "notes"
        self.settings = settings
        self.store = store

    def write(self, name: str, text: str) -> Path:
        path = self.notes / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class _InlineThread:
    """Runs the background update to completion inside the request, so tests can check it."""

    def __init__(self, target: Callable[..., None], args: tuple[object, ...] = (), **_: object):
        self._target, self._args = target, args

    def start(self) -> None:
        self._target(*self._args)


@pytest.fixture
def library(
    tmp_path: Path,
    session: Session,
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Library]:
    # pytest's temp folders live under AppData, which real folders may not; use a stand-in home.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    notes = tmp_path / "notes"
    notes.mkdir()
    settings = make_settings(allowed_folders=[notes], embedding_model=MODEL)
    embedder = HashingEmbedder(model_name=MODEL)
    store = QdrantVectorStore.in_memory("library-test", embedder.dimension)

    @contextmanager
    def locked(*_args: object, **_kwargs: object) -> Iterator[QdrantVectorStore]:
        yield store

    monkeypatch.setattr(library_sync, "locked_vector_store", locked)
    monkeypatch.setattr(library_sync, "load_embedder", lambda *_a, **_k: embedder)
    monkeypatch.setattr(library_sync, "is_model_downloaded", lambda *_a, **_k: True)
    monkeypatch.setattr(library_sync.threading, "Thread", _InlineThread)
    monkeypatch.setattr(library_sync, "_status", library_sync.SyncStatus())
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = lambda: session
    result = Library(tmp_path, settings, store)
    result.write("oats.md", "# Oats\n\nOats are high in fibre.\n")
    result.write("rice.txt", "Rice needs twice its volume of water.")
    yield result
    store.close()
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _sync(client: TestClient) -> dict[str, object]:
    response = client.post("/api/v1/library/sync")
    assert response.status_code == 202
    body: dict[str, object] = client.get("/api/v1/library/sync").json()
    return body


def _documents(client: TestClient) -> list[dict[str, object]]:
    body = client.get("/api/v1/library/documents").json()
    documents: list[dict[str, object]] = body["documents"]
    return documents


# --- updating the library ---------------------------------------------------------------------


def test_an_update_reads_the_folder_and_makes_the_notes_searchable(
    client: TestClient, library: Library
) -> None:
    status = _sync(client)

    assert (status["state"], status["added"], status["indexed"]) == ("done", 2, 2)
    documents = _documents(client)
    assert [d["name"] for d in documents] == ["oats.md", "rice.txt"]
    assert all(d["searchable"] for d in documents)
    assert library.store.count() > 0


def test_each_document_says_where_it_came_from(client: TestClient, library: Library) -> None:
    _sync(client)

    sources = {d["name"]: d["source"] for d in _documents(client)}

    assert Path(str(sources["oats.md"])) == (library.notes / "oats.md").resolve()


def test_a_second_update_adds_nothing(client: TestClient, library: Library) -> None:
    _sync(client)

    status = _sync(client)

    assert (status["added"], status["unchanged"], status["indexed"]) == (0, 2, 0)


def test_an_update_while_another_runs_is_refused(client: TestClient, library: Library) -> None:
    library_sync._status.state = "running"

    assert client.post("/api/v1/library/sync").status_code == 409


def test_an_update_reports_a_missing_model_instead_of_failing(
    client: TestClient, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(library_sync, "is_model_downloaded", lambda *_a, **_k: False)

    status = _sync(client)

    assert status["state"] == "done"
    assert "download-model" in str(status["message"])
    assert not any(d["searchable"] for d in _documents(client))


def test_an_update_that_breaks_reports_a_fixed_sentence_not_details(
    client: TestClient, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret note text should never appear")

    monkeypatch.setattr(library_sync, "ingest_folders", explode)

    status = _sync(client)

    assert status["state"] == "failed"
    assert "secret" not in str(status["message"])


# --- removing documents -----------------------------------------------------------------------


def test_removing_a_document_deletes_it_and_its_vectors_but_not_your_file(
    client: TestClient, library: Library, session: Session
) -> None:
    _sync(client)
    oats = next(d for d in _documents(client) if d["name"] == "oats.md")
    assert library.store.count(int(str(oats["id"]))) > 0
    document = session.get(Document, oats["id"])
    assert document is not None
    stored_copy = library.settings.data_dir / document.stored_path
    assert stored_copy.exists()

    response = client.delete(f"/api/v1/library/documents/{oats['id']}")

    assert response.status_code == 204
    assert [d["name"] for d in _documents(client)] == ["rice.txt"]
    assert library.store.count(int(str(oats["id"]))) == 0
    assert (library.notes / "oats.md").exists()
    assert not stored_copy.exists()


def test_a_removed_document_stays_out_until_you_restore_it(
    client: TestClient, library: Library
) -> None:
    _sync(client)
    oats = next(d for d in _documents(client) if d["name"] == "oats.md")
    client.delete(f"/api/v1/library/documents/{oats['id']}")

    status = _sync(client)
    assert status["skipped_excluded"] == 1
    assert [d["name"] for d in _documents(client)] == ["rice.txt"]
    assert client.get("/api/v1/library/documents").json()["removed_count"] == 1

    assert client.post("/api/v1/library/removed/restore").json() == {"restored": 1}
    _sync(client)
    assert [d["name"] for d in _documents(client)] == ["oats.md", "rice.txt"]


def test_removing_an_unknown_document_is_a_404(client: TestClient, library: Library) -> None:
    assert client.delete("/api/v1/library/documents/999").status_code == 404


def test_the_library_file_holds_paths_and_hashes_but_no_note_text(
    client: TestClient, library: Library
) -> None:
    _sync(client)

    stored = (library.settings.data_dir / LIBRARY_FILENAME).read_text(encoding="utf-8")

    assert "high in fibre" not in stored
    assert "twice its volume" not in stored
    assert set(json.loads(stored)) == {"folders", "sources", "excluded"}


# --- folders ----------------------------------------------------------------------------------


def test_the_folder_list_marks_where_each_folder_came_from(
    client: TestClient, library: Library
) -> None:
    _sync(client)

    folders = client.get("/api/v1/library/folders").json()["folders"]

    assert len(folders) == 1
    assert folders[0]["origin"] == "env"
    assert folders[0]["removable"] is False
    assert folders[0]["exists"] is True
    assert folders[0]["documents"] == 2


def test_a_folder_can_be_added_and_then_removed(
    client: TestClient, library: Library, tmp_path: Path
) -> None:
    extra = tmp_path / "extra"
    extra.mkdir()

    added = client.post("/api/v1/library/folders", json={"path": str(extra)})

    assert added.status_code == 201
    entries = {Path(f["path"]).name: f for f in added.json()["folders"]}
    assert entries["extra"]["origin"] == "ui"
    assert entries["extra"]["removable"] is True

    removed = client.delete("/api/v1/library/folders", params={"path": str(extra)})
    assert removed.status_code == 200
    assert [Path(f["path"]).name for f in removed.json()["folders"]] == ["notes"]


def test_a_folder_added_in_the_interface_is_read_by_the_next_update(
    client: TestClient, library: Library, tmp_path: Path
) -> None:
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "trip.md").write_text("# Trip\n\nThe flight leaves at seven.\n", encoding="utf-8")
    client.post("/api/v1/library/folders", json={"path": str(extra)})

    _sync(client)

    assert "trip.md" in [d["name"] for d in _documents(client)]


def test_removing_a_folder_can_remove_its_documents_too(
    client: TestClient, library: Library, tmp_path: Path
) -> None:
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "trip.md").write_text("# Trip\n\nThe flight leaves at seven.\n", encoding="utf-8")
    client.post("/api/v1/library/folders", json={"path": str(extra)})
    _sync(client)

    result = client.delete(
        "/api/v1/library/folders", params={"path": str(extra), "remove_documents": "true"}
    ).json()

    assert result["documents_removed"] == 1
    assert [d["name"] for d in _documents(client)] == ["oats.md", "rice.txt"]
    # Not marked as removed: adding the folder again brings the notes back.
    assert client.get("/api/v1/library/documents").json()["removed_count"] == 0


def test_a_folder_from_the_env_file_cannot_be_removed_here(
    client: TestClient, library: Library
) -> None:
    response = client.delete("/api/v1/library/folders", params={"path": str(library.notes)})

    assert response.status_code == 422
    assert ".env" in response.json()["detail"]


def test_removing_a_folder_that_is_not_listed_is_refused(
    client: TestClient, library: Library, tmp_path: Path
) -> None:
    response = client.delete("/api/v1/library/folders", params={"path": str(tmp_path / "nope")})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("make_path", "expected"),
    [
        (lambda tmp: str(tmp / "missing"), "does not exist"),
        (lambda tmp: "relative/folder", "full path"),
        (lambda tmp: "   ", "full path"),
        (lambda tmp: str(tmp / "notes" / "oats.md"), "not a folder"),
        (lambda tmp: str(Path(tmp.anchor)), "whole drive"),
        (lambda tmp: str(Path.home()), "user folder"),
        (lambda tmp: str(tmp), "own data"),
        (lambda tmp: str(tmp / "data"), "own data"),
        (lambda tmp: str(tmp / "notes"), "already on the list"),
    ],
)
def test_unsafe_or_pointless_folders_are_refused(
    client: TestClient,
    library: Library,
    tmp_path: Path,
    make_path: Callable[[Path], str],
    expected: str,
) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)

    response = client.post("/api/v1/library/folders", json={"path": make_path(tmp_path)})

    assert response.status_code == 422
    assert expected in response.json()["detail"]


@pytest.mark.skipif(not os.environ.get("WINDIR"), reason="Windows only")
def test_the_windows_system_folder_is_refused(client: TestClient, library: Library) -> None:
    response = client.post("/api/v1/library/folders", json={"path": os.environ["WINDIR"]})

    assert response.status_code == 422
    assert "System folders" in response.json()["detail"]


def test_a_folder_that_holds_keys_is_refused(
    client: TestClient, library: Library, tmp_path: Path
) -> None:
    keys = tmp_path / ".ssh"
    keys.mkdir()

    response = client.post("/api/v1/library/folders", json={"path": str(keys)})

    assert response.status_code == 422
    assert "keys" in response.json()["detail"]


# --- edited and deleted files -------------------------------------------------------------------


def test_an_edited_file_replaces_its_old_version_in_the_list_and_the_index(
    client: TestClient, library: Library, session: Session
) -> None:
    _sync(client)
    old_id = next(d["id"] for d in _documents(client) if d["name"] == "rice.txt")
    assert library.store.count(old_id) > 0
    library.write("rice.txt", "Rice needs twice its volume of water and eighteen minutes.")

    status = _sync(client)

    assert (status["replaced"], status["missing"], status["added"]) == (1, 0, 1)
    assert [d["name"] for d in _documents(client)] == ["oats.md", "rice.txt"]  # not three
    assert old_id not in {d["id"] for d in _documents(client)}
    assert library.store.count(old_id) == 0
    assert session.get(Document, old_id) is not None  # kept as history


def test_a_deleted_file_is_flagged_missing_after_two_updates_and_leaves_the_search(
    client: TestClient, library: Library
) -> None:
    _sync(client)
    (library.notes / "rice.txt").unlink()

    first = _sync(client)
    second = _sync(client)

    assert (first["missing"], second["missing"]) == (0, 1)
    rice = next(d for d in _documents(client) if d["name"] == "rice.txt")
    assert (rice["status"], rice["searchable"]) == ("missing", False)
    assert library.store.count(rice["id"]) == 0  # type: ignore[arg-type]
    oats = next(d for d in _documents(client) if d["name"] == "oats.md")
    assert (oats["status"], oats["searchable"]) == ("active", True)


def test_the_library_file_no_longer_keeps_the_path_of_every_document(
    client: TestClient, library: Library
) -> None:
    _sync(client)

    stored = json.loads((library.settings.data_dir / LIBRARY_FILENAME).read_text(encoding="utf-8"))

    assert stored["sources"] == {}
    assert "oats.md" not in json.dumps(stored)  # the places live, encrypted, in the database


def test_the_folder_counts_come_from_the_recorded_places(
    client: TestClient, library: Library
) -> None:
    _sync(client)

    folders = client.get("/api/v1/library/folders").json()["folders"]

    assert [f["documents"] for f in folders] == [2]


def test_removing_a_document_removes_the_older_versions_of_its_file(
    client: TestClient, library: Library, session: Session
) -> None:
    _sync(client)
    library.write("rice.txt", "Rice needs twice its volume of water and eighteen minutes.")
    _sync(client)
    current = next(d for d in _documents(client) if d["name"] == "rice.txt")

    assert client.delete(f"/api/v1/library/documents/{current['id']}").status_code == 204

    names = [d.original_filename for d in session.query(Document).all()]
    assert names == ["oats.md"]


def test_the_update_reports_its_progress_in_chunks_too(
    client: TestClient, library: Library
) -> None:
    status = _sync(client)

    assert (status["chunks_total"], status["chunks_done"]) == (2, 2)


# --- looking at one document ----------------------------------------------------------------------


def _by_name(client: TestClient, name: str) -> dict[str, object]:
    return next(d for d in _documents(client) if d["name"] == name)


def test_a_document_can_be_inspected_with_its_places_and_search_state(
    client: TestClient, library: Library
) -> None:
    _sync(client)
    oats = _by_name(client, "oats.md")

    detail = client.get(f"/api/v1/library/documents/{oats['id']}").json()

    assert detail["name"] == "oats.md" and detail["status"] == "active"
    assert detail["searchable"] is True and detail["indexed_chunks"] == detail["chunks"] > 0
    assert detail["supersedes_id"] is None and detail["older_versions"] == []
    place = detail["locations"][0]
    assert Path(place["folder"]) == library.notes.resolve() and place["path"] == "oats.md"
    assert (place["status"], place["misses"]) == ("present", 0)


def test_an_edited_document_shows_its_earlier_versions(
    client: TestClient, library: Library
) -> None:
    _sync(client)
    first_id = _by_name(client, "rice.txt")["id"]
    library.write("rice.txt", "Rice needs twice its volume of water and eighteen minutes.")
    _sync(client)
    library.write("rice.txt", "Rice needs twice its volume of water and twenty minutes.")
    _sync(client)
    current = _by_name(client, "rice.txt")

    detail = client.get(f"/api/v1/library/documents/{current['id']}").json()

    assert len(detail["older_versions"]) == 2
    assert detail["older_versions"][-1]["id"] == first_id  # the oldest is last
    assert detail["supersedes_id"] == detail["older_versions"][0]["id"]
    assert all(v["chunks"] >= 1 for v in detail["older_versions"])


def test_a_missing_file_shows_how_many_updates_missed_it(
    client: TestClient, library: Library
) -> None:
    _sync(client)
    (library.notes / "rice.txt").unlink()
    _sync(client)
    _sync(client)
    rice = _by_name(client, "rice.txt")

    detail = client.get(f"/api/v1/library/documents/{rice['id']}").json()

    assert detail["status"] == "missing"
    assert (detail["locations"][0]["status"], detail["locations"][0]["misses"]) == ("missing", 2)


def test_an_unknown_document_is_a_404_in_every_inspection_endpoint(
    client: TestClient, library: Library
) -> None:
    for path in ("/documents/999", "/documents/999/chunks", "/jobs/999"):
        assert client.get(f"/api/v1/library{path}").status_code == 404, path


def test_the_passages_of_a_document_can_be_paged(client: TestClient, library: Library) -> None:
    library.write("long.md", "\n\n".join(f"Paragraph {n} " + "word " * 120 for n in range(12)))
    _sync(client)
    long = _by_name(client, "long.md")

    first = client.get(f"/api/v1/library/documents/{long['id']}/chunks?limit=3").json()
    second = client.get(f"/api/v1/library/documents/{long['id']}/chunks?limit=3&offset=3").json()

    assert first["total"] == long["chunks"] > 6
    assert [c["index"] for c in first["chunks"]] == [0, 1, 2]
    assert [c["index"] for c in second["chunks"]] == [3, 4, 5]
    chunk = first["chunks"][0]
    assert chunk["text"].startswith("Paragraph 0") and chunk["searchable"] is True
    assert chunk["start_page"] is None and chunk["char_count"] == len(chunk["text"])


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "limit=x"])
def test_chunk_paging_rejects_silly_values(
    client: TestClient, library: Library, query: str
) -> None:
    _sync(client)
    oats = _by_name(client, "oats.md")

    response = client.get(f"/api/v1/library/documents/{oats['id']}/chunks?{query}")

    assert response.status_code == 422


# --- listing with pages and a filter -------------------------------------------------------------


def test_the_list_can_be_paged_and_says_how_many_there_are_in_all(
    client: TestClient, library: Library
) -> None:
    _sync(client)

    everything = client.get("/api/v1/library/documents").json()
    first = client.get("/api/v1/library/documents?limit=1").json()
    second = client.get("/api/v1/library/documents?limit=1&offset=1").json()
    beyond = client.get("/api/v1/library/documents?limit=5&offset=5").json()

    assert everything["total"] == first["total"] == second["total"] == 2
    assert [d["name"] for d in everything["documents"]] == ["oats.md", "rice.txt"]
    assert [d["name"] for d in first["documents"]] == ["oats.md"]
    assert [d["name"] for d in second["documents"]] == ["rice.txt"]
    assert beyond["documents"] == [] and beyond["total"] == 2


def test_the_list_can_be_filtered_by_whether_the_file_is_still_found(
    client: TestClient, library: Library
) -> None:
    _sync(client)
    (library.notes / "rice.txt").unlink()
    _sync(client)
    _sync(client)

    active = client.get("/api/v1/library/documents?state=active").json()
    missing = client.get("/api/v1/library/documents?state=missing").json()

    assert [d["name"] for d in active["documents"]] == ["oats.md"] and active["total"] == 1
    assert [d["name"] for d in missing["documents"]] == ["rice.txt"] and missing["total"] == 1
    assert client.get("/api/v1/library/documents?state=other").status_code == 422


# --- the history of updates ----------------------------------------------------------------------


def test_every_update_is_recorded_with_its_counts(client: TestClient, library: Library) -> None:
    first = _sync(client)
    library.write("rice.txt", "Rice needs twice its volume of water and twenty minutes.")
    second = _sync(client)

    jobs = client.get("/api/v1/library/jobs").json()

    assert [j["id"] for j in jobs] == [second["job_id"], first["job_id"]]  # newest first
    assert jobs[1]["state"] == "done" and (jobs[1]["added"], jobs[1]["indexed"]) == (2, 2)
    assert (jobs[0]["added"], jobs[0]["replaced"]) == (1, 1)
    assert jobs[0]["finished_at"] is not None and jobs[0]["kind"] == "sync"
    one = client.get(f"/api/v1/library/jobs/{first['job_id']}").json()
    assert one == jobs[1]


def test_a_failed_update_is_recorded_with_its_sentence(
    client: TestClient, library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_a: object, **_k: object) -> None:
        raise RuntimeError("private detail: /home/me/secret.txt")

    monkeypatch.setattr(library_sync, "ingest_folders", broken)

    status = _sync(client)
    job = client.get(f"/api/v1/library/jobs/{status['job_id']}").json()

    assert job["state"] == "failed"
    assert "unexpected error" in job["message"] and "secret" not in json.dumps(job)


def test_a_run_cut_short_by_a_restart_is_shown_as_interrupted(
    client: TestClient, library: Library, session: Session
) -> None:
    from app.knowledge.library.jobs import start_job

    orphan = start_job(session)  # as if the program stopped while it was running

    job = client.get(f"/api/v1/library/jobs/{orphan}").json()

    assert job["state"] == "interrupted" and "stopped while" in job["message"]
    assert job["finished_at"] is not None


def test_a_run_that_is_really_running_is_not_called_interrupted(
    client: TestClient, library: Library, session: Session
) -> None:
    from app.knowledge.library.jobs import start_job

    running = start_job(session)
    library_sync._status.state = "running"
    library_sync._status.job_id = running

    job = client.get(f"/api/v1/library/jobs/{running}").json()

    assert job["state"] == "running" and job["finished_at"] is None


def test_only_the_most_recent_updates_are_kept(session: Session) -> None:
    from app.knowledge.library import jobs

    for _ in range(jobs.MAX_JOBS_KEPT + 5):
        jobs.finish_job(session, jobs.start_job(session), "done", None, {"added": 1})

    kept = jobs.list_jobs(session, limit=500)
    assert len(kept) == jobs.MAX_JOBS_KEPT
    assert kept[0].id > kept[-1].id and kept[0].counts["added"] == 1


def test_the_history_holds_no_names_or_paths(client: TestClient, library: Library) -> None:
    status = _sync(client)

    stored = json.dumps(client.get(f"/api/v1/library/jobs/{status['job_id']}").json())

    assert "oats" not in stored and "rice" not in stored and str(library.notes) not in stored
