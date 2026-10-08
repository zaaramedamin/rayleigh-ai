"""keep a log of what the agent did and was asked to do

Every request the agent makes for a tool is written here, whether it was allowed, refused or sent to
the owner for approval, together with the owner's answer and what came of it. The free text
(`detail`: the arguments with long text cut and secrets hidden, or a short summary of a result) uses
the encrypted type and is listed in `app/security/migrate.py`.

The log is append-only: a trigger makes the database refuse any change to the facts of a row.
Only `detail` can be rewritten, because encrypting a library rewrites it in place. The owner can
erase the log as a whole, never edit it.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("tool", sa.String(length=40), nullable=True),
        sa.Column("level", sa.String(length=20), nullable=True),
        sa.Column("decision", sa.String(length=10), nullable=True),
        sa.Column("decided_by", sa.String(length=10), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
    )
    op.create_index("ix_agent_events_created_at", "agent_events", ["created_at"])
    op.create_index("ix_agent_events_run_id", "agent_events", ["run_id"])
    op.execute(
        "CREATE TRIGGER IF NOT EXISTS agent_events_no_update "
        "BEFORE UPDATE OF created_at, run_id, step, kind, tool, level, decision, "
        "decided_by ON agent_events "
        "BEGIN SELECT RAISE(ABORT, 'the agent log cannot be changed'); END"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS agent_events_no_update")
    op.drop_index("ix_agent_events_run_id", table_name="agent_events")
    op.drop_index("ix_agent_events_created_at", table_name="agent_events")
    op.drop_table("agent_events")
