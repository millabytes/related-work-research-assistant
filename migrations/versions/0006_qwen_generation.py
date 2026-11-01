"""Allow Qwen generations while preserving historical Gemma rows."""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_related_work_generations_ck_related_work_generations_model_key"),
        "related_work_generations",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_related_work_generations_model_key"),
        "related_work_generations",
        "model_key IN ('primera','gemma','qwen')",
    )
    op.drop_constraint(
        op.f("ck_related_work_jobs_ck_related_work_jobs_model_key"),
        "related_work_jobs",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_related_work_jobs_model_key"),
        "related_work_jobs",
        "model_key IN ('primera','gemma','qwen')",
    )


def downgrade() -> None:
    # Qwen rows cannot satisfy the prior schema. Gemma and PRIMERA history remain.
    op.execute("DELETE FROM related_work_jobs WHERE model_key = 'qwen'")
    op.execute("DELETE FROM related_work_generations WHERE model_key = 'qwen'")
    op.execute("DELETE FROM model_versions WHERE model_key LIKE 'generation:qwen:%'")
    op.drop_constraint(
        op.f("ck_related_work_jobs_model_key"), "related_work_jobs", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_related_work_jobs_model_key"),
        "related_work_jobs",
        "model_key IN ('primera','gemma')",
    )
    op.drop_constraint(
        op.f("ck_related_work_generations_model_key"),
        "related_work_generations",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_related_work_generations_model_key"),
        "related_work_generations",
        "model_key IN ('primera','gemma')",
    )
