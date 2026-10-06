"""A record of library updates, kept in the database so a restart still shows what happened.

A row holds counts and a short fixed sentence, never file names, paths or note text.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.storage.models import Job

KIND_SYNC = "sync"
STATE_RUNNING = "running"
INTERRUPTED_MESSAGE = "The program stopped while this update was running."
MAX_JOBS_KEPT = 100

COUNTERS = (
    "added",
    "unchanged",
    "replaced",
    "missing",
    "skipped_excluded",
    "failed_files",
    "indexed",
    "chunks_done",
)


@dataclass(frozen=True)
class JobInfo:
    id: int
    kind: str
    state: str
    message: str | None
    started_at: datetime
    finished_at: datetime | None
    counts: Mapping[str, int]


def _info(job: Job) -> JobInfo:
    return JobInfo(
        id=job.id,
        kind=job.kind,
        state=job.state,
        message=job.message,
        started_at=job.started_at,
        finished_at=job.finished_at,
        counts={name: getattr(job, name) for name in COUNTERS},
    )


def start_job(session: Session, kind: str = KIND_SYNC) -> int:
    job = Job(kind=kind, state=STATE_RUNNING, started_at=datetime.now(UTC))
    session.add(job)
    session.commit()
    return job.id


def finish_job(
    session: Session, job_id: int, state: str, message: str | None, counts: Mapping[str, int]
) -> None:
    job = session.get(Job, job_id)
    if job is None:
        return
    job.state = state
    job.message = message
    job.finished_at = datetime.now(UTC)
    for name in COUNTERS:
        setattr(job, name, int(counts.get(name, 0)))
    session.commit()
    _forget_old(session)


def _forget_old(session: Session) -> None:
    """Keep the most recent runs only."""
    ids = session.scalars(select(Job.id).order_by(Job.id.desc()).offset(MAX_JOBS_KEPT)).all()
    if ids:
        for job in session.scalars(select(Job).where(Job.id.in_(ids))):
            session.delete(job)
        session.commit()


def mark_interrupted(session: Session, *, running_job_id: int | None) -> int:
    """Runs still marked as running although no update is running were cut short by a restart."""
    query = update(Job).where(Job.state == STATE_RUNNING)
    if running_job_id is not None:
        query = query.where(Job.id != running_job_id)
    result = session.execute(
        query.values(
            state="interrupted", message=INTERRUPTED_MESSAGE, finished_at=datetime.now(UTC)
        )
    )
    session.commit()
    return int(result.rowcount)  # type: ignore[attr-defined]  # an UPDATE result has a row count


def list_jobs(session: Session, *, limit: int = 20) -> list[JobInfo]:
    jobs = session.scalars(select(Job).order_by(Job.id.desc()).limit(limit)).all()
    return [_info(job) for job in jobs]


def get_job(session: Session, job_id: int) -> JobInfo | None:
    job = session.get(Job, job_id)
    return None if job is None else _info(job)
