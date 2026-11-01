"""Allow separately versioned Qwen LoRA generation jobs and results."""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_related_work_generations_model_key"),
        "related_work_generations",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_related_work_generations_model_key"),
        "related_work_generations",
        "model_key IN ('primera','gemma','qwen','qwen_lora')",
    )
    op.drop_constraint(
        op.f("ck_related_work_jobs_model_key"), "related_work_jobs", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_related_work_jobs_model_key"),
        "related_work_jobs",
        "model_key IN ('primera','gemma','qwen','qwen_lora')",
    )


def downgrade() -> None:
    op.execute("DELETE FROM related_work_jobs WHERE model_key = 'qwen_lora'")
    op.execute("DELETE FROM related_work_generations WHERE model_key = 'qwen_lora'")
    op.execute("DELETE FROM model_versions WHERE model_key LIKE 'generation:qwen_lora:%'")
    op.drop_constraint(
        op.f("ck_related_work_jobs_model_key"), "related_work_jobs", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_related_work_jobs_model_key"),
        "related_work_jobs",
        "model_key IN ('primera','gemma','qwen')",
    )
    op.drop_constraint(
        op.f("ck_related_work_generations_model_key"),
        "related_work_generations",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_related_work_generations_model_key"),
        "related_work_generations",
        "model_key IN ('primera','gemma','qwen')",
    )
