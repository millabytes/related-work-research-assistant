"""Store generation finish reasons and validation failures."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_related_work_generations_ck_related_work_generations_status"),
        "related_work_generations",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_related_work_generations_status"),
        "related_work_generations",
        "status IN ('completed','context_overflow','invalid','failed')",
    )
    op.add_column(
        "related_work_generations", sa.Column("finish_reason", sa.String(24), nullable=True)
    )
    op.add_column(
        "related_work_generations",
        sa.Column(
            "validation_errors",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_related_work_generations_status"),
        "related_work_generations",
        type_="check",
    )
    op.execute("UPDATE related_work_generations SET status = 'failed' WHERE status = 'invalid'")
    op.create_check_constraint(
        op.f("ck_related_work_generations_status"),
        "related_work_generations",
        "status IN ('completed','context_overflow','failed')",
    )
    op.drop_column("related_work_generations", "validation_errors")
    op.drop_column("related_work_generations", "finish_reason")
