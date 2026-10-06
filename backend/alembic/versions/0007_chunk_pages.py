"""remember which pages each chunk comes from

For formats that have pages (PDF), so that a citation can say "page 12". Both columns stay empty
for plain text and the other formats without pages.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("start_page", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("end_page", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("chunks") as batch:
        batch.drop_column("end_page")
        batch.drop_column("start_page")
