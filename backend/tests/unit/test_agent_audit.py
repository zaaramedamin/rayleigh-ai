"""The agent's audit log: what is recorded, how it is protected, and how it fails."""

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.orm import Session

from app.agent.audit import (
    HIDDEN,
    INTERRUPTED_DETAIL,
    MAX_ARGUMENT_CHARS,
    MAX_DETAIL_CHARS,
    AuditError,
    all_events,
    close_unfinished_runs,
    erase_log,
    new_run_id,
    recent_events,
    record,
    redact_arguments,
    run_events,
    summarize_result,
)
from app.security.migrate import ENCRYPTED_COLUMNS
from app.storage.models import AgentEvent

# --- what an entry holds ---------------------------------------------------------------------


def test_an_entry_keeps_what_was_asked_and_who_decided(session: Session) -> None:
    run = new_run_id()

    entry = record(
        session,
        run,
        "decision",
        step=2,
        tool="open_path",
        level="open_local",
        decision="ask",
        decided_by="policy",
        detail='{"path": "C:/notes.txt"}',
    )

    assert entry.id and entry.created_at is not None
    stored = run_events(session, run)[0]
    assert (stored.kind, stored.step, stored.tool, stored.level) == (
        "decision",
        2,
        "open_path",
        "open_local",
    )
    assert (stored.decision, stored.decided_by, stored.detail) == (
        "ask",
        "policy",
        '{"path": "C:/notes.txt"}',
    )


def test_a_run_id_is_unique_and_fits_the_column() -> None:
    ids = {new_run_id() for _ in range(50)}

    assert len(ids) == 50 and all(len(i) == 32 for i in ids)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"kind": "deleted"}, "an entry is one of"),
        ({"kind": "decision", "decision": "maybe"}, "a decision is one of"),
        ({"kind": "approval", "decided_by": "model"}, "made by one of"),
    ],
)
def test_an_entry_with_an_unknown_kind_decision_or_decider_is_refused(
    session: Session, fields: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        record(session, new_run_id(), **fields)
    assert recent_events(session) == []


def test_the_detail_is_cut_to_its_limit(session: Session) -> None:
    entry = record(session, new_run_id(), "result", detail="x" * (MAX_DETAIL_CHARS * 3))

    assert entry.detail is not None and len(entry.detail) < MAX_DETAIL_CHARS + 60
    assert "more characters" in entry.detail


# --- redaction -------------------------------------------------------------------------------


def test_long_text_in_arguments_is_cut_and_says_how_much_was_left_out() -> None:
    shown = json.loads(redact_arguments({"text": "a" * 1000, "short": "ok"}))

    assert shown["short"] == "ok"
    assert (
        shown["text"].startswith("a" * MAX_ARGUMENT_CHARS)
        and "(+800 more characters)" in shown["text"]
    )


@pytest.mark.parametrize(
    "name",
    [
        "password",
        "api_key",
        "Token",
        "client_secret",
        "cookie",
        "Authorization",
        "bearer",
        "credentials",
    ],
)
def test_anything_that_looks_like_a_secret_is_hidden(name: str) -> None:
    shown = json.loads(
        redact_arguments({name: "hunter2-very-secret", "url": "https://example.org"})
    )

    assert shown[name] == HIDDEN and shown["url"] == "https://example.org"
    assert "hunter2" not in redact_arguments({name: "hunter2-very-secret"})


def test_other_kinds_of_value_are_kept_or_written_out_and_cut() -> None:
    shown = json.loads(
        redact_arguments(
            {"count": 3, "ratio": 0.5, "on": True, "nothing": None, "list": list(range(500))}
        )
    )

    assert (shown["count"], shown["ratio"], shown["on"], shown["nothing"]) == (3, 0.5, True, None)
    assert isinstance(shown["list"], str) and "more characters" in shown["list"]


def test_redaction_is_stable_and_never_longer_than_the_detail_limit() -> None:
    many = {f"field{i}": "y" * 150 for i in range(60)}

    text = redact_arguments(many)

    assert len(text) < MAX_DETAIL_CHARS + 60
    assert redact_arguments({"b": 1, "a": 2}) == redact_arguments({"a": 2, "b": 1})


def test_a_result_is_summarised_on_one_line_and_never_kept_in_full() -> None:
    summary = summarize_result("line one\n\n  line   two\t" + "z" * 1000)

    assert summary.startswith("line one line two zzz") and "\n" not in summary
    assert len(summary) < 400 and summary.endswith("more characters)")
    assert summarize_result("  short  ") == "short"


# --- reading ---------------------------------------------------------------------------------


def test_one_task_reads_back_in_order_and_apart_from_the_others(session: Session) -> None:
    first, second = new_run_id(), new_run_id()
    for kind in ("run_started", "request", "decision"):
        record(session, first, kind)
        record(session, second, "run_started")

    assert [e.kind for e in run_events(session, first)] == ["run_started", "request", "decision"]
    assert len(run_events(session, second)) == 3


def test_recent_entries_come_newest_first_and_are_limited(session: Session) -> None:
    run = new_run_id()
    for step in range(5):
        record(session, run, "request", step=step)

    assert [e.step for e in recent_events(session, limit=3)] == [4, 3, 2]


def test_an_export_reads_everything_oldest_first_in_batches(session: Session) -> None:
    run = new_run_id()
    for step in range(7):
        record(session, run, "request", step=step)

    assert [e.step for e in all_events(session, batch=3)] == list(range(7))


# --- it cannot be edited, only erased --------------------------------------------------------


def test_the_database_refuses_to_change_an_entry(session: Session) -> None:
    entry = record(session, new_run_id(), "decision", decision="deny", decided_by="policy")

    with pytest.raises(DatabaseError, match="cannot be changed"):
        session.execute(
            update(AgentEvent).where(AgentEvent.id == entry.id).values(decision="allow")
        )
    session.rollback()

    assert session.get(AgentEvent, entry.id).decision == "deny"  # type: ignore[union-attr]


def test_the_owner_can_erase_the_whole_log(session: Session) -> None:
    run = new_run_id()
    for _ in range(4):
        record(session, run, "request")

    assert erase_log(session) == 4
    assert recent_events(session) == [] and erase_log(session) == 0


# --- it fails loudly -------------------------------------------------------------------------


def test_an_entry_that_cannot_be_written_raises_so_the_caller_does_not_act(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> None:
        raise OperationalError("INSERT", {}, Exception("disk full"))

    monkeypatch.setattr(session, "commit", broken)

    with pytest.raises(AuditError, match="could not be written"):
        record(session, new_run_id(), "request")


# --- privacy ---------------------------------------------------------------------------------


def test_only_counts_reach_the_application_log(
    session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        record(session, new_run_id(), "request", tool="open_path", detail="Quillon-Marmalade-4821")
        erase_log(session)

    assert "agent event kind=request decision=None" in caplog.text
    assert "agent log erased entries=1" in caplog.text
    assert "Quillon-Marmalade-4821" not in caplog.text


def test_the_free_text_is_listed_for_encryption() -> None:
    assert ("agent_events", "detail", "agent_events.detail") in ENCRYPTED_COLUMNS


# --- tasks the program left unfinished --------------------------------------------------------


def started(session: Session, run_id: str, ago: timedelta, *, finished: bool = False) -> None:
    """A task as the log would hold it, begun `ago` ago."""
    when = datetime.now(UTC) - ago
    session.add(AgentEvent(run_id=run_id, kind="run_started", created_at=when))
    if finished:
        session.add(AgentEvent(run_id=run_id, kind="run_finished", created_at=when))
    session.commit()


def test_a_task_the_program_left_unfinished_is_closed_in_the_log(session: Session) -> None:
    started(session, "old", timedelta(hours=3))

    assert close_unfinished_runs(session) == 1

    events = run_events(session, "old")
    assert [e.kind for e in events] == ["run_started", "run_finished"]
    assert events[-1].detail == INTERRUPTED_DETAIL and "interrupted" in INTERRUPTED_DETAIL


def test_only_old_unfinished_tasks_are_touched(session: Session) -> None:
    started(session, "finished-long-ago", timedelta(hours=5), finished=True)
    started(session, "running-right-now", timedelta(minutes=2))
    started(session, "stale-one", timedelta(hours=2))
    started(session, "stale-two", timedelta(days=3))

    assert close_unfinished_runs(session) == 2

    assert [e.kind for e in run_events(session, "finished-long-ago")] == [
        "run_started",
        "run_finished",
    ]
    assert [e.kind for e in run_events(session, "running-right-now")] == ["run_started"]
    for run_id in ("stale-one", "stale-two"):
        assert [e.kind for e in run_events(session, run_id)] == ["run_started", "run_finished"]


def test_closing_is_done_once_and_an_empty_log_is_fine(session: Session) -> None:
    assert close_unfinished_runs(session) == 0
    started(session, "old", timedelta(hours=3))

    assert close_unfinished_runs(session) == 1
    assert close_unfinished_runs(session) == 0
    assert len(run_events(session, "old")) == 2


def test_how_long_a_task_may_look_unfinished_can_be_chosen(session: Session) -> None:
    started(session, "recent", timedelta(minutes=10))

    assert close_unfinished_runs(session) == 0
    assert close_unfinished_runs(session, older_than=timedelta(minutes=5)) == 1


def test_closing_logs_only_a_count(session: Session, caplog: pytest.LogCaptureFixture) -> None:
    started(session, "old", timedelta(hours=3))

    with caplog.at_level(logging.DEBUG):
        close_unfinished_runs(session)

    assert "agent tasks closed after an interruption count=1" in caplog.text
