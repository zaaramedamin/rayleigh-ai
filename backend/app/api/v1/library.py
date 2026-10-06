from dataclasses import asdict
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.api import library_sync
from app.api.deps import SessionDep, SettingsDep
from app.knowledge.library import documents as library_documents
from app.knowledge.library import folders as library_folders
from app.knowledge.library import jobs as library_jobs
from app.knowledge.library.state import load_state
from app.storage.models import KIND_PROFILE, Chunk, Document

router = APIRouter(prefix="/library", tags=["library"])


class DocumentOut(BaseModel):
    id: int
    name: str
    size_bytes: int
    media_type: str
    created_at: datetime
    chunks: int
    searchable: bool
    source: str | None = Field(description="Where the file was read from, if known.")
    is_profile: bool
    status: Literal["active", "missing"] = Field(
        description='"missing" when the file was not found in the last two updates.'
    )


class DocumentList(BaseModel):
    documents: list[DocumentOut]
    removed_count: int = Field(description="Documents you removed that scans keep out.")
    total: int = Field(description="Documents that match the filter, before paging.")


class LocationOut(BaseModel):
    folder: str
    path: str = Field(description="Inside the folder, with forward slashes.")
    status: Literal["present", "missing"]
    last_seen_at: datetime
    misses: int = Field(description="Updates in a row that did not find the file.")


class VersionOut(BaseModel):
    id: int
    created_at: datetime
    size_bytes: int
    chunks: int


class DocumentDetail(DocumentOut):
    supersedes_id: int | None
    locations: list[LocationOut]
    older_versions: list[VersionOut] = Field(
        description="Earlier versions of this file, newest first. Kept as history, never searched."
    )
    indexed_chunks: int


class ChunkOut(BaseModel):
    index: int
    heading_path: str
    start_line: int
    end_line: int
    start_page: int | None
    end_page: int | None
    char_count: int
    text: str
    searchable: bool


class ChunkPage(BaseModel):
    total: int
    chunks: list[ChunkOut]


class JobOut(BaseModel):
    id: int
    kind: str
    state: Literal["running", "done", "failed", "interrupted"]
    message: str | None
    started_at: datetime
    finished_at: datetime | None
    added: int
    unchanged: int
    replaced: int
    missing: int
    skipped_excluded: int
    failed_files: int
    indexed: int
    chunks_done: int


class FolderOut(BaseModel):
    path: str
    origin: Literal["env", "ui"]
    exists: bool
    removable: bool
    documents: int = Field(description="Documents known to come from this folder.")


class FolderList(BaseModel):
    folders: list[FolderOut]


class NewFolder(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


class FolderRemoval(FolderList):
    documents_removed: int


class SyncOut(BaseModel):
    state: Literal["idle", "running", "done", "failed"]
    phase: str
    message: str | None
    started_at: str | None
    finished_at: str | None
    folders: int
    added: int
    unchanged: int
    skipped_excluded: int
    failed_files: int
    replaced: int = Field(description="Documents replaced by a newer version of their file.")
    missing: int = Field(description="Documents whose file was not found again.")
    indexing_total: int
    indexed: int
    chunks_total: int
    chunks_done: int
    job_id: int | None = Field(description="This run's record in the job history.")


def _folder_list(settings: SettingsDep, session: SessionDep) -> FolderList:
    entries = library_folders.allowed_folders(settings)
    counts = library_documents.documents_per_folder(
        session, settings.data_dir, [entry.path for entry in entries]
    )
    return FolderList(
        folders=[
            FolderOut(
                path=str(entry.path),
                origin=entry.origin,
                exists=entry.path.resolve().is_dir(),
                removable=entry.origin == "ui",
                documents=counts.get(entry.path, 0),
            )
            for entry in entries
        ]
    )


def _document_out(info: library_documents.DocumentInfo) -> DocumentOut:
    return DocumentOut(
        id=info.id,
        name=info.name,
        size_bytes=info.size_bytes,
        media_type=info.media_type,
        created_at=info.created_at,
        chunks=info.chunks,
        searchable=info.searchable,
        source=info.source,
        is_profile=info.kind == KIND_PROFILE,
        status="missing" if info.status == "missing" else "active",
    )


@router.get("/documents")
def list_documents(
    session: SessionDep,
    settings: SettingsDep,
    state: Literal["active", "missing"] | None = Query(
        default=None, description="Only documents whose file is still found, or no longer is."
    ),
    limit: int | None = Query(default=None, ge=1, le=1000, description="At most this many."),
    offset: int = Query(default=0, ge=0, description="Skip this many (sorted by name)."),
) -> DocumentList:
    """Everything in the library, with where each file came from. Without `limit`, all of it."""
    infos = library_documents.list_documents(session, settings.data_dir, settings.embedding_model)
    if state is not None:
        infos = [i for i in infos if (i.status == "missing") == (state == "missing")]
    total = len(infos)
    page = infos[offset : offset + limit] if limit is not None else infos[offset:]
    return DocumentList(
        documents=[_document_out(i) for i in page],
        removed_count=len(load_state(settings.data_dir).excluded),
        total=total,
    )


@router.get("/documents/{document_id}")
def get_document(document_id: int, session: SessionDep, settings: SettingsDep) -> DocumentDetail:
    """One document: where it was found, its earlier versions and how much of it is searchable."""
    details = library_documents.document_details(
        session, settings.data_dir, document_id, settings.embedding_model
    )
    if details is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That document is not in the library.")
    base = _document_out(details.info)
    return DocumentDetail(
        **base.model_dump(),
        supersedes_id=details.supersedes_id,
        locations=[
            LocationOut(
                folder=loc.folder,
                path=loc.path,
                status="missing" if loc.status == "missing" else "present",
                last_seen_at=loc.last_seen_at,
                misses=loc.misses,
            )
            for loc in details.locations
        ],
        older_versions=[
            VersionOut(id=v.id, created_at=v.created_at, size_bytes=v.size_bytes, chunks=v.chunks)
            for v in details.older_versions
        ],
        indexed_chunks=details.indexed_chunks,
    )


@router.get("/documents/{document_id}/chunks")
def get_document_chunks(
    document_id: int,
    session: SessionDep,
    settings: SettingsDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ChunkPage:
    """The passages a document was cut into, in order, so you can see what the assistant sees."""
    if session.get(Document, document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That document is not in the library.")
    total = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    rows = session.scalars(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .offset(offset)
        .limit(limit)
    ).all()
    return ChunkPage(
        total=total or 0,
        chunks=[
            ChunkOut(
                index=c.chunk_index,
                heading_path=c.heading_path,
                start_line=c.start_line,
                end_line=c.end_line,
                start_page=c.start_page,
                end_page=c.end_page,
                char_count=c.char_count,
                text=c.text,
                searchable=c.indexed_model == settings.embedding_model,
            )
            for c in rows
        ],
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, session: SessionDep, settings: SettingsDep) -> None:
    """Remove a document from the library. The original file in your folder is not touched."""
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That document is not in the library.")
    library_sync.delete_vectors(settings, [document_id])
    library_documents.delete_document(session, settings.data_dir, document, exclude=True)


@router.post("/removed/restore")
def restore_removed(settings: SettingsDep) -> dict[str, int]:
    """Let the next update add back the documents you removed."""
    return {"restored": library_documents.clear_exclusions(settings.data_dir)}


@router.get("/folders")
def list_folders(session: SessionDep, settings: SettingsDep) -> FolderList:
    return _folder_list(settings, session)


@router.post("/folders", status_code=status.HTTP_201_CREATED)
def add_folder(folder: NewFolder, session: SessionDep, settings: SettingsDep) -> FolderList:
    """Allow Reyleight to read one more folder. It is read when you run an update."""
    try:
        library_folders.add_folder(settings, folder.path)
    except library_folders.FolderError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _folder_list(settings, session)


@router.delete("/folders")
def remove_folder(
    session: SessionDep,
    settings: SettingsDep,
    path: str = Query(min_length=1, max_length=1000),
    remove_documents: bool = False,
) -> FolderRemoval:
    """Stop reading a folder, and optionally remove the documents that came from it."""
    try:
        target = library_folders.remove_folder(settings, path)
    except library_folders.FolderError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    removed = 0
    if remove_documents:
        stale = library_documents.documents_from_folder(session, settings.data_dir, target)
        library_sync.delete_vectors(settings, [d.id for d in stale])
        for document in stale:
            library_documents.delete_document(session, settings.data_dir, document, exclude=False)
            removed += 1
    return FolderRemoval(folders=_folder_list(settings, session).folders, documents_removed=removed)


def _job_out(job: library_jobs.JobInfo) -> JobOut:
    return JobOut(
        id=job.id,
        kind=job.kind,
        state=job.state,  # type: ignore[arg-type]  # one of the four states the table holds
        message=job.message,
        started_at=job.started_at,
        finished_at=job.finished_at,
        **job.counts,
    )


@router.get("/jobs")
def list_jobs(session: SessionDep, limit: int = Query(default=20, ge=1, le=100)) -> list[JobOut]:
    """The most recent library updates, newest first, including those from before a restart."""
    current = library_sync.current_status()
    running_id = current.job_id if current.state == "running" else None
    library_jobs.mark_interrupted(session, running_job_id=running_id)
    return [_job_out(job) for job in library_jobs.list_jobs(session, limit=limit)]


@router.get("/jobs/{job_id}")
def get_job(job_id: int, session: SessionDep) -> JobOut:
    current = library_sync.current_status()
    running_id = current.job_id if current.state == "running" else None
    library_jobs.mark_interrupted(session, running_job_id=running_id)
    job = library_jobs.get_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "There is no such update.")
    return _job_out(job)


@router.get("/sync")
def sync_status() -> SyncOut:
    return _sync_out()


@router.post("/sync", status_code=status.HTTP_202_ACCEPTED)
def start_sync(settings: SettingsDep) -> SyncOut:
    """Read the folders and make anything new searchable. Progress is on GET /library/sync."""
    if not library_sync.start_sync(settings):
        raise HTTPException(status.HTTP_409_CONFLICT, "An update is already running.")
    return _sync_out()


def _sync_out() -> SyncOut:
    return SyncOut.model_validate(asdict(library_sync.current_status()))
