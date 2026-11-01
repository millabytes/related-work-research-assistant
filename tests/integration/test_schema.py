import pytest
from alembic import command
from conftest import migration_config
from sqlalchemy import inspect, text

from related_work_assistant.database.base import Base

pytestmark = pytest.mark.integration


def test_schema_and_migration_roundtrip(db_engine):
    with db_engine.begin() as conn:
        cfg = migration_config(conn)
        command.upgrade(cfg, "head")
        inspector = inspect(conn)
        assert set(inspector.get_table_names()) == set(Base.metadata.tables) | {"alembic_version"}
        for name, table in Base.metadata.tables.items():
            assert {c["name"] for c in inspector.get_columns(name)} == set(table.columns.keys())
            assert len(inspector.get_foreign_keys(name)) == len(table.foreign_key_constraints)
        assert conn.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
        assert (
            conn.scalar(
                text(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid='chunk_embeddings'::regclass AND attname='embedding'"
                )
            )
            == "vector"
        )
        command.check(cfg)
        command.downgrade(cfg, "base")
        assert inspect(conn).get_table_names() == ["alembic_version"]
        assert conn.scalar(text("SELECT 1 FROM pg_extension WHERE extname='vector'")) == 1
        command.upgrade(cfg, "head")
