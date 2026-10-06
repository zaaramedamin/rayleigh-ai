"""Slice 2.1: where documents come from, edited files, missing files, and prune."""

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.knowledge.indexing.service import (
    count_pending,
    count_stale_vectors,
    index_pending,
    library_counts,
)
from app.knowledge.ingestion.service import IngestSummary, ingest_folders
from app.knowledge.ingestion.sync import MASS_DISAPPEARANCE_MIN
from app.knowledge.library.documents import (
    delete_document,
    documents_from_folder,
    documents_per_folder,
    list_documents,
)
from app.knowledge.library.prune import find_candidates, prune
from app.knowledge.retrieval.service import retrieve
from app.storage.files import resolve_inside
from app.storage.models import (
    DOC_ACTIVE,
    DOC_MISSING,
    DOC_SUPERSEDED,
    LOCATION_MISSING,
    LOCATION_PRESENT,
    Chunk,
    Document,
    DocumentSource,
)
from app.storage.vector_store import QdrantVectorStore
from tests.fakes import HashingEmbedder

MAX_BYTES = 100_000
MODEL = "test/hashing-embedder"


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    folder = tmp_path / "notes"
    folder.mkdir()
    return folder


def sync(
    session: Session,
    data_dir: Path,
    *folders: Path,
    legacy: dict[str, str] | None = None,
) -> IngestSummary:
    return ingest_folders(session, data_dir, list(folders), MAX_BYTES, legacy_sources=legacy)


def states(session: Session) -> list[tuple[str, str]]:
    """(name, status) of every document, oldest first."""
    rows = session.scalars(select(Document).order_by(Document.id)).all()
    return [(d.original_filename, d.status) for d in rows]


def places(session: Session) -> list[tuple[str, str, int]]:
    """(path, status, misses) of every location."""
    rows = session.scalars(select(DocumentSource).order_by(DocumentSource.id)).all()
    return [(r.source_path, r.status, r.misses) for r in rows]


def count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


# --- 2.1a: locations are recorded --------------------------------------------------------------


def test_every_file_found_is_recorded_with_its_place(
    session: Session, data_dir: Path, notes: Path
) -> None:
    (notes / "a.txt").write_text("alpha")
    (notes / "sub").mkdir()
    (notes / "sub" / "b.md").write_text("# Beta")

    first = sync(session, data_dir, notes)
    second = sync(session, data_dir, notes)

    rows = session.scalars(select(DocumentSource).order_by(DocumentSource.source_path)).all()
    assert [(r.source_root, r.source_path) for r in rows] == [
        (str(notes.resolve()), "a.txt"),
        (str(notes.resolve()), "sub/b.md"),  # forward slashes on every system
    ]
    assert first.sync.locations_added == 2
    assert second.sync.locations_added == 0
    assert count(session, Document) == 2


def test_a_file_belongs_to_the_most_specific_allowed_folder(
    session: Session, data_dir: Path, notes: Path
) -> None:
    inner = notes / "inner"
    inner.mkdir()
    (inner / "x.txt").write_text("x")

    sync(session, data_dir, notes, inner)

    row = session.scalars(select(DocumentSource)).one()
    assert (row.source_root, row.source_path) == (str(inner.resolve()), "x.txt")


def test_the_same_content_in_two_folders_is_one_document_with_two_places(
    session: Session, data_dir: Path, notes: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (notes / "a.txt").write_text("same words")
    (other / "copy.txt").write_text("same words")

    sync(session, data_dir, notes, other)

    assert count(session, Document) == 1
    assert sorted(path for path, _status, _misses in places(session)) == ["a.txt", "copy.txt"]


# --- 2.1b: an edit replaces the old version ----------------------------------------------------


def test_an_edited_file_replaces_its_old_version(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("version one")
    sync(session, data_dir, notes)
    old = session.scalars(select(Document)).one()
    note.write_text("version two")

    summary = sync(session, data_dir, notes)

    assert states(session) == [("a.txt", DOC_SUPERSEDED), ("a.txt", DOC_ACTIVE)]
    new = session.scalars(select(Document).where(Document.status == DOC_ACTIVE)).one()
    assert new.supersedes_id == old.id
    assert summary.sync.superseded == 1
    assert summary.sync.superseded_ids == [old.id]
    # The place now points at the new version, and nothing was added as a second place.
    assert session.scalars(select(DocumentSource)).one().document_id == new.id
    # The old text is kept as history.
    assert count(session, Chunk) == 2


def test_changing_a_file_back_brings_the_old_version_back(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("version one")
    sync(session, data_dir, notes)
    note.write_text("version two")
    sync(session, data_dir, notes)
    note.write_text("version one")

    summary = sync(session, data_dir, notes)

    assert states(session) == [("a.txt", DOC_ACTIVE), ("a.txt", DOC_SUPERSEDED)]
    assert summary.sync.reactivated == 1
    assert count(session, Document) == 2  # no third document for the same words


def test_an_edit_does_not_retire_a_version_that_is_still_found_elsewhere(
    session: Session, data_dir: Path, notes: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (notes / "a.txt").write_text("shared")
    (other / "a.txt").write_text("shared")
    sync(session, data_dir, notes, other)
    (notes / "a.txt").write_text("edited here only")

    summary = sync(session, data_dir, notes, other)

    assert states(session) == [("a.txt", DOC_ACTIVE), ("a.txt", DOC_ACTIVE)]
    assert summary.sync.superseded == 0


def test_after_an_edit_only_the_new_text_is_searched(
    session: Session, data_dir: Path, notes: Path
) -> None:
    embedder = HashingEmbedder()
    store = QdrantVectorStore.in_memory("sync-edit", embedder.dimension)
    note = notes / "trip.txt"
    note.write_text("The flight leaves at seven from gate BANANA")
    sync(session, data_dir, notes)
    index_pending(session, embedder, store)
    old = session.scalars(select(Document)).one()
    assert store.count(old.id) > 0
    note.write_text("The flight leaves at nine from gate CHERRY")
    sync(session, data_dir, notes)

    # Before any re-index the old vectors still exist, but they are never returned.
    assert store.count(old.id) > 0
    assert retrieve(session, embedder, store, "flight gate banana", top_k=5) == []

    summary = index_pending(session, embedder, store)

    assert summary.documents_purged == 1
    assert store.count(old.id) == 0
    results = retrieve(session, embedder, store, "flight gate banana", top_k=5)
    assert results
    assert all("BANANA" not in r.text for r in results)
    assert any("CHERRY" in r.text for r in results)
    store.close()


def test_replaced_versions_are_not_indexed_again_and_are_counted(
    session: Session, data_dir: Path, notes: Path
) -> None:
    embedder = HashingEmbedder(model_name=MODEL)
    store = QdrantVectorStore.in_memory("sync-count", embedder.dimension)
    note = notes / "a.txt"
    note.write_text("one")
    sync(session, data_dir, notes)
    index_pending(session, embedder, store)
    note.write_text("two")
    sync(session, data_dir, notes)

    assert (count_pending(session, MODEL), count_stale_vectors(session)) == (1, 1)
    index_pending(session, embedder, store)

    counts = library_counts(session, MODEL)
    assert (counts.active, counts.superseded, counts.missing) == (1, 1, 0)
    assert (counts.pending, counts.searchable, count_stale_vectors(session)) == (0, 1, 0)
    assert counts.chunks == 1  # of the current version only
    store.close()


# --- 2.1c: a file that stays away is marked missing, carefully ---------------------------------


def test_a_deleted_file_is_marked_missing_only_after_two_updates(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("alpha")
    sync(session, data_dir, notes)
    note.unlink()

    first = sync(session, data_dir, notes)
    assert states(session) == [("a.txt", DOC_ACTIVE)]
    assert places(session) == [("a.txt", LOCATION_PRESENT, 1)]
    assert first.sync.newly_missing == 0

    second = sync(session, data_dir, notes)
    assert states(session) == [("a.txt", DOC_MISSING)]
    assert places(session) == [("a.txt", LOCATION_MISSING, 2)]
    assert second.sync.newly_missing == 1

    third = sync(session, data_dir, notes)  # nothing new to say
    assert third.sync.newly_missing == 0


def test_a_file_that_comes_back_makes_its_document_active_again(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("alpha")
    sync(session, data_dir, notes)
    note.unlink()
    sync(session, data_dir, notes)
    sync(session, data_dir, notes)
    assert states(session) == [("a.txt", DOC_MISSING)]
    note.write_text("alpha")

    summary = sync(session, data_dir, notes)

    assert states(session) == [("a.txt", DOC_ACTIVE)]
    assert places(session) == [("a.txt", LOCATION_PRESENT, 0)]
    assert summary.sync.reactivated == 1


def test_one_missed_update_is_forgiven_when_the_file_returns(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("alpha")
    sync(session, data_dir, notes)
    note.rename(notes / "hidden.tmp")
    sync(session, data_dir, notes)
    (notes / "hidden.tmp").rename(note)

    sync(session, data_dir, notes)

    assert places(session) == [("a.txt", LOCATION_PRESENT, 0)]


def test_a_renamed_file_is_the_same_document_at_a_new_place(
    session: Session, data_dir: Path, notes: Path
) -> None:
    (notes / "a.txt").write_text("alpha")
    sync(session, data_dir, notes)
    (notes / "a.txt").rename(notes / "b.txt")

    sync(session, data_dir, notes)
    sync(session, data_dir, notes)

    assert count(session, Document) == 1
    assert states(session) == [("a.txt", DOC_ACTIVE)]  # still found, at its new place
    assert sorted(places(session)) == [
        ("a.txt", LOCATION_MISSING, 2),
        ("b.txt", LOCATION_PRESENT, 0),
    ]


def test_a_folder_that_cannot_be_read_is_not_judged(
    session: Session, data_dir: Path, tmp_path: Path
) -> None:
    drive = tmp_path / "usb"
    drive.mkdir()
    (drive / "a.txt").write_text("alpha")
    sync(session, data_dir, drive)
    drive.rename(tmp_path / "usb-unplugged")

    summaries = [sync(session, data_dir, drive) for _ in range(3)]

    assert summaries[-1].folders_missing == 1
    assert states(session) == [("a.txt", DOC_ACTIVE)]
    assert places(session) == [("a.txt", LOCATION_PRESENT, 0)]


def test_an_emptied_file_is_not_a_missing_file(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("alpha")
    sync(session, data_dir, notes)
    note.write_text("")  # it exists, it just has nothing in it

    for _ in range(3):
        sync(session, data_dir, notes)

    assert states(session) == [("a.txt", DOC_ACTIVE)]
    assert places(session) == [("a.txt", LOCATION_PRESENT, 0)]


def test_many_files_vanishing_at_once_are_not_judged(
    session: Session, data_dir: Path, notes: Path
) -> None:
    total = MASS_DISAPPEARANCE_MIN + 1
    for number in range(total):
        (notes / f"n{number}.txt").write_text(f"note {number}")
    sync(session, data_dir, notes)
    for number in range(MASS_DISAPPEARANCE_MIN):
        (notes / f"n{number}.txt").unlink()

    summaries = [sync(session, data_dir, notes) for _ in range(3)]

    assert all(summary.sync.paused_folders == 1 for summary in summaries)
    assert {status for _name, status in states(session)} == {DOC_ACTIVE}
    assert all(misses == 0 for _path, _status, misses in places(session))


def test_a_few_vanished_files_in_a_big_folder_are_judged_normally(
    session: Session, data_dir: Path, notes: Path
) -> None:
    for number in range(20):
        (notes / f"n{number}.txt").write_text(f"note {number}")
    sync(session, data_dir, notes)
    (notes / "n0.txt").unlink()
    (notes / "n1.txt").unlink()

    sync(session, data_dir, notes)
    summary = sync(session, data_dir, notes)

    assert summary.sync.newly_missing == 2
    assert sum(1 for _name, status in states(session) if status == DOC_MISSING) == 2


def test_documents_read_before_places_were_recorded_are_followed_too(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("alpha")
    sync(session, data_dir, notes)
    document = session.scalars(select(Document)).one()
    session.query(DocumentSource).delete()  # as if it was read before this feature existed
    session.commit()
    legacy = {document.content_hash: str(note.resolve())}
    note.unlink()

    first = sync(session, data_dir, notes, legacy=legacy)
    assert states(session) == [("a.txt", DOC_ACTIVE)]
    assert places(session) == [("a.txt", LOCATION_PRESENT, 1)]
    second = sync(session, data_dir, notes, legacy=legacy)

    assert (first.sync.newly_missing, second.sync.newly_missing) == (0, 1)
    assert states(session) == [("a.txt", DOC_MISSING)]
    assert count(session, DocumentSource) == 1  # the old place was not added a second time


# --- listing and removing ----------------------------------------------------------------------


def test_the_library_list_hides_old_versions_and_flags_missing_files(
    session: Session, data_dir: Path, notes: Path
) -> None:
    (notes / "edited.txt").write_text("one")
    (notes / "gone.txt").write_text("keep me")
    sync(session, data_dir, notes)
    (notes / "edited.txt").write_text("two")
    (notes / "gone.txt").unlink()
    sync(session, data_dir, notes)
    sync(session, data_dir, notes)

    listed = list_documents(session, data_dir, MODEL)

    assert [(d.name, d.status) for d in listed] == [
        ("edited.txt", DOC_ACTIVE),
        ("gone.txt", DOC_MISSING),
    ]
    assert listed[0].source == str(notes.resolve() / "edited.txt")


def test_documents_are_counted_and_found_per_folder(
    session: Session, data_dir: Path, notes: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (notes / "a.txt").write_text("alpha")
    (notes / "b.txt").write_text("beta")
    (other / "c.txt").write_text("gamma")
    sync(session, data_dir, notes, other)
    (notes / "a.txt").write_text("alpha edited")
    sync(session, data_dir, notes, other)

    counts = documents_per_folder(session, data_dir, [notes, other])

    assert counts == {notes: 2, other: 1}  # the replaced version is not counted
    assert {d.original_filename for d in documents_from_folder(session, data_dir, notes)} == {
        "a.txt",
        "b.txt",
    }


def test_removing_a_note_also_removes_its_older_versions(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("first words")
    sync(session, data_dir, notes)
    note.write_text("second words")
    sync(session, data_dir, notes)
    documents = session.scalars(select(Document).order_by(Document.id)).all()
    stored = [resolve_inside(data_dir, d.stored_path) for d in documents]
    assert all(path.exists() for path in stored)

    removed = delete_document(session, data_dir, documents[1], exclude=False)

    assert (removed.documents, removed.chunks, removed.stored_files) == (2, 2, 2)
    assert count(session, Document) == count(session, Chunk) == count(session, DocumentSource) == 0
    assert not any(path.exists() for path in stored)


# --- 2.1d: prune --------------------------------------------------------------------------------


def _make_one_missing_and_one_replaced(
    session: Session, data_dir: Path, notes: Path
) -> tuple[Path, Path]:
    gone, edited = notes / "gone.txt", notes / "edited.txt"
    gone.write_text("this file will vanish")
    edited.write_text("first draft")
    sync(session, data_dir, notes)
    gone.unlink()
    edited.write_text("second draft")
    sync(session, data_dir, notes)
    sync(session, data_dir, notes)
    return gone, edited


def test_looking_for_what_to_prune_changes_nothing(
    session: Session, data_dir: Path, notes: Path
) -> None:
    _make_one_missing_and_one_replaced(session, data_dir, notes)
    before = states(session)

    found = find_candidates(session, include_superseded=False)

    assert [d.original_filename for d in found.missing] == ["gone.txt"]
    assert found.superseded == []
    assert states(session) == before


def test_prune_removes_a_missing_document_everywhere(
    session: Session, data_dir: Path, notes: Path
) -> None:
    embedder = HashingEmbedder(model_name=MODEL)
    store = QdrantVectorStore.in_memory("prune", embedder.dimension)
    _make_one_missing_and_one_replaced(session, data_dir, notes)
    index_pending(session, embedder, store)
    gone = session.scalars(select(Document).where(Document.status == DOC_MISSING)).one()
    gone_id, stored = gone.id, resolve_inside(data_dir, gone.stored_path)
    assert stored.exists()

    removed = prune(session, data_dir, find_candidates(session, include_superseded=False), store)

    assert (removed.documents, removed.stored_files) == (1, 1)
    assert session.get(Document, gone_id) is None
    assert (
        session.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == gone_id))
        == 0
    )
    assert (
        session.scalar(
            select(func.count())
            .select_from(DocumentSource)
            .where(DocumentSource.document_id == gone_id)
        )
        == 0
    )
    assert not stored.exists()
    assert store.count(gone_id) == 0
    # The replaced version and the current one are untouched.
    assert sorted(states(session)) == [
        ("edited.txt", DOC_ACTIVE),
        ("edited.txt", DOC_SUPERSEDED),
    ]
    store.close()


def test_prune_leaves_old_versions_unless_asked(
    session: Session, data_dir: Path, notes: Path
) -> None:
    _make_one_missing_and_one_replaced(session, data_dir, notes)

    prune(session, data_dir, find_candidates(session, include_superseded=False), None)
    assert (DOC_SUPERSEDED in {s for _n, s in states(session)}) is True

    removed = prune(session, data_dir, find_candidates(session, include_superseded=True), None)

    assert removed.documents == 1
    assert states(session) == [("edited.txt", DOC_ACTIVE)]


def test_pruning_a_replaced_version_with_its_missing_successor_does_not_fail(
    session: Session, data_dir: Path, notes: Path
) -> None:
    note = notes / "a.txt"
    note.write_text("v1")
    sync(session, data_dir, notes)
    note.write_text("v2")
    sync(session, data_dir, notes)
    note.unlink()
    sync(session, data_dir, notes)
    sync(session, data_dir, notes)
    assert sorted(states(session)) == [("a.txt", DOC_MISSING), ("a.txt", DOC_SUPERSEDED)]

    removed = prune(session, data_dir, find_candidates(session, include_superseded=True), None)

    assert removed.documents == 2
    assert count(session, Document) == 0
