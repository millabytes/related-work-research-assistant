"""Persist measured generation latency and peak accelerator memory."""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "related_work_generations",
        sa.Column("inference_latency_ms", sa.Float(), nullable=True),
    )
    op.add_column(
        "related_work_generations",
        sa.Column("peak_memory_mb", sa.Float(), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_related_work_generations_inference_latency_ms"),
        "related_work_generations",
        "inference_latency_ms >= 0",
    )
    op.create_check_constraint(
        op.f("ck_related_work_generations_peak_memory_mb"),
        "related_work_generations",
        "peak_memory_mb >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_related_work_generations_peak_memory_mb"),
        "related_work_generations",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_related_work_generations_inference_latency_ms"),
        "related_work_generations",
        type_="check",
    )
    op.drop_column("related_work_generations", "peak_memory_mb")
    op.drop_column("related_work_generations", "inference_latency_ms")
