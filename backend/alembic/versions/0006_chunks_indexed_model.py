"""remember, per chunk, which embedding model has its vector

An interrupted index run can then continue at the first chunk without a vector instead of starting
the document again. Existing chunks take the model of their document, which is complete.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("indexed_model", sa.String(length=255), nullable=True))
    op.execute(
        "UPDATE chunks SET indexed_model = "
        "(SELECT documents.indexed_model FROM documents WHERE documents.id = chunks.document_id)"
    )


def downgrade() -> None:
    with op.batch_alter_table("chunks") as batch:
        batch.drop_column("indexed_model")
