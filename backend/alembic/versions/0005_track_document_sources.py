"""track where documents come from, and which version is current

Adds document_sources (one row per place a file was found), and to documents: status (active,
superseded, missing), supersedes_id (the version an edit replaced) and kind (note or profile).
Existing documents become active notes, and the profile note is marked as such.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The name the profile note is saved under (app.knowledge.profile.service.PROFILE_FILENAME).
# Written out here on purpose: a migration must keep working when the application changes.
PROFILE_NOTE_NAME = "My profile.md"


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("status", sa.String(length=12), nullable=False, server_default="active"),
    )
    op.add_column("documents", sa.Column("supersedes_id", sa.Integer(), nullable=True))
    op.add_column(
        "documents",
        sa.Column("kind", sa.String(length=12), nullable=False, server_default="note"),
    )
    op.create_table(
        "document_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_root", sa.String(length=1000), nullable=False),
        sa.Column("source_path", sa.String(length=1000), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False, server_default="present"),
        sa.Column("misses", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("missing_since", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_document_sources_document_id", "document_sources", ["document_id"])

    # Works on a library that is not encrypted. In an encrypted one the name is unreadable here;
    # such a library was encrypted after this migration existed, so the profile is marked when it
    # is saved (see save_profile).
    op.get_bind().execute(
        sa.text("UPDATE documents SET kind = 'profile' WHERE original_filename = :name"),
        {"name": PROFILE_NOTE_NAME},
    )


def downgrade() -> None:
    op.drop_index("ix_document_sources_document_id", table_name="document_sources")
    op.drop_table("document_sources")
    with op.batch_alter_table("documents") as batch:
        batch.drop_column("kind")
        batch.drop_column("supersedes_id")
        batch.drop_column("status")
