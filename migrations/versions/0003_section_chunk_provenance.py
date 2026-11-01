"""Add full-text provenance to paper sections.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("paper_sections", sa.Column("source_file_id", sa.Uuid(), nullable=True))
    op.add_column("paper_sections", sa.Column("source_sha256", sa.CHAR(64), nullable=True))
    op.add_column("paper_sections", sa.Column("parser_version", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_paper_sections_source_file_id_paper_files",
        "paper_sections",
        "paper_files",
        ["source_file_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_paper_sections_source_file_id", "paper_sections", ["source_file_id"], unique=False
    )
    op.create_check_constraint(
        "ck_paper_sections_source_sha256",
        "paper_sections",
        "source_sha256 IS NULL OR source_sha256 ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.drop_constraint("ck_paper_sections_source_sha256", "paper_sections", type_="check")
    op.drop_index("ix_paper_sections_source_file_id", table_name="paper_sections")
    op.drop_constraint(
        "fk_paper_sections_source_file_id_paper_files", "paper_sections", type_="foreignkey"
    )
    op.drop_column("paper_sections", "parser_version")
    op.drop_column("paper_sections", "source_sha256")
    op.drop_column("paper_sections", "source_file_id")
