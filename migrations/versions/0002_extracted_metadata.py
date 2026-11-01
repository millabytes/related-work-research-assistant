"""Preserve structured author names and unknown titles from extraction."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("papers", "title", existing_type=sa.Text(), nullable=True)
    op.add_column("paper_authors", sa.Column("first_name", sa.Text(), nullable=True))
    op.add_column("paper_authors", sa.Column("last_name", sa.Text(), nullable=True))


def downgrade() -> None:
    # Refuse to fabricate titles or discard rows if unknown titles were ingested.
    if op.get_bind().scalar(sa.text("SELECT count(*) FROM papers WHERE title IS NULL")):
        raise RuntimeError("Resolve papers with unknown titles before downgrading to 0001")
    op.drop_column("paper_authors", "last_name")
    op.drop_column("paper_authors", "first_name")
    op.alter_column("papers", "title", existing_type=sa.Text(), nullable=False)
