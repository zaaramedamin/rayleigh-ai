"""The agent's audit log: a local, append-only record of everything it was asked to do.

Every request for a tool is written here, including the ones that were refused, together with the
decision (by the policy or by the owner) and what came of it. The log is built so it can be trusted
and so it does not become a copy of the owner's private notes:

- Arguments are recorded with long text cut and anything that looks like a secret hidden, and a
  result is recorded as a short summary, never in full.
- The free text is stored in an encrypted column when the library is encrypted.
- Rows are never edited (the database refuses it). There is no function here that changes a row; the
  owner can only erase the log as a whole.
- Recording can fail, and then it says so (AuditError). The agent loop treats that as a reason
  not to act: nothing runs without a record that it was asked for.
- Only counts are written to the application log, never what the entries say.
"""

import json
import logging
import re
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.storage.models import AgentEvent

logger = logging.getLogger(__name__)

EVENT_KINDS = (
    "run_started",
    "request",
    "decision",
    "approval",
    "result",
    "problem",
    "run_finished",
)
DECISIONS = ("allow", "ask", "deny")
DECIDED_BY = ("policy", "owner")

MAX_ARGUMENT_CHARS = 200
MAX_SUMMARY_CHARS = 300
MAX_DETAIL_CHARS = 1500
HIDDEN = "[hidden]"
_SECRET_NAME = re.compile(r"pass|secret|token|key|credential|cookie|auth|bearer", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


class AuditError(RuntimeError):
    """An entry could not be recorded. Whatever it was about must not go ahead."""


def new_run_id() -> str:
    return uuid.uuid4().hex


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}... (+{len(text) - limit} more characters)"


def redact_arguments(arguments: Mapping[str, Any]) -> str:
    """The arguments of a request as JSON for the log: long text cut, secrets hidden."""
    shown: dict[str, Any] = {}
    for name, value in arguments.items():
        key = str(name)
        if _SECRET_NAME.search(key):
            shown[key] = HIDDEN
        elif isinstance(value, str):
            shown[key] = _cut(value, MAX_ARGUMENT_CHARS)
        elif isinstance(value, bool | int | float) or value is None:
            shown[key] = value
        else:
            shown[key] = _cut(
                json.dumps(value, ensure_ascii=False, default=str), MAX_ARGUMENT_CHARS
            )
    return _cut(json.dumps(shown, ensure_ascii=False, sort_keys=True), MAX_DETAIL_CHARS)


def summarize_result(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """The start of a result on one line, for the log. A result is never recorded in full."""
    return _cut(_WHITESPACE.sub(" ", text).strip(), limit)


def record(
    session: Session,
    run_id: str,
    kind: str,
    *,
    step: int = 0,
    tool: str | None = None,
    level: str | None = None,
    decision: str | None = None,
    decided_by: str | None = None,
    detail: str | None = None,
) -> AgentEvent:
    """Write one entry and commit it. Raises AuditError if it cannot be written."""
    if kind not in EVENT_KINDS:
        raise ValueError(f"an entry is one of: {', '.join(EVENT_KINDS)}")
    if decision is not None and decision not in DECISIONS:
        raise ValueError(f"a decision is one of: {', '.join(DECISIONS)}")
    if decided_by is not None and decided_by not in DECIDED_BY:
        raise ValueError(f"a decision is made by one of: {', '.join(DECIDED_BY)}")
    entry = AgentEvent(
        run_id=run_id,
        step=step,
        kind=kind,
        tool=tool,
        level=level,
        decision=decision,
        decided_by=decided_by,
        detail=_cut(detail, MAX_DETAIL_CHARS) if detail is not None else None,
    )
    try:
        session.add(entry)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise AuditError("The agent's log could not be written.") from exc
    logger.info("agent event kind=%s decision=%s", kind, decision)
    return entry


def run_events(session: Session, run_id: str) -> list[AgentEvent]:
    """Every entry of one task, in the order it happened."""
    return list(
        session.scalars(
            select(AgentEvent).where(AgentEvent.run_id == run_id).order_by(AgentEvent.id)
        )
    )


def recent_events(session: Session, limit: int = 100) -> list[AgentEvent]:
    """The latest entries across all tasks, newest first."""
    return list(session.scalars(select(AgentEvent).order_by(AgentEvent.id.desc()).limit(limit)))


def all_events(session: Session, batch: int = 500) -> Iterator[AgentEvent]:
    """Every entry, oldest first, for an export."""
    last_id = 0
    while True:
        rows = list(
            session.scalars(
                select(AgentEvent)
                .where(AgentEvent.id > last_id)
                .order_by(AgentEvent.id)
                .limit(batch)
            )
        )
        if not rows:
            return
        yield from rows
        last_id = rows[-1].id


def erase_log(session: Session) -> int:
    """Erase the whole log (the owner's choice) and say how many entries went."""
    try:
        removed = session.scalar(select(func.count()).select_from(AgentEvent)) or 0
        session.execute(delete(AgentEvent))
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise AuditError("The agent's log could not be erased.") from exc
    logger.info("agent log erased entries=%d", removed)
    return removed
