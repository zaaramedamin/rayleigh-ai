"""A removed note's text must not stay readable inside the database file."""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.ingestion.service import ingest_folders
from app.knowledge.library.documents import delete_document
from app.storage.database import DB_FILENAME, Base, create_db_engine, vacuum_database
from app.storage.models import Document

SECRET = b"Quillon-Marmalade-Zeppelin-4821"


def test_a_removed_notes_text_is_gone_from_the_database_file(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "diary.txt").write_text(f"My secret is {SECRET.decode()}.")
    data_dir = tmp_path / "data"
    engine = create_db_engine(data_dir)
    Base.metadata.create_all(engine)
    database_file = data_dir / DB_FILENAME

    with Session(engine) as session:
        ingest_folders(session, data_dir, [notes], 100_000)
        assert SECRET in database_file.read_bytes(), "the test needs the text to be there first"

        document = session.scalars(select(Document)).one()
        delete_document(session, data_dir, document, exclude=False)

    assert SECRET not in database_file.read_bytes()  # overwritten as soon as it was deleted
    vacuum_database(engine)
    assert SECRET not in database_file.read_bytes()
    engine.dispose()
