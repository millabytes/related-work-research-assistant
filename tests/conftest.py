from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from related_work_assistant.core.config import PROJECT_ROOT, Settings


def migration_config(connection):
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["connection"] = connection
    return config


@pytest.fixture(scope="session")
def db_engine():
    """Create/drop only a uniquely named database owned by this test session."""
    try:
        url = Settings().sqlalchemy_url
    except ValueError:
        pytest.skip("Integration tests require configured local PostgreSQL")
    admin = create_engine(
        url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 3},
    )
    name = "related_work_test_" + uuid4().hex
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception:
        admin.dispose()
        pytest.fail(
            "Cannot create temporary PostgreSQL database; start Docker and check credentials"
        )
    engine = create_engine(url.set(database=name), hide_parameters=True)
    try:
        with engine.begin() as conn:
            command.upgrade(migration_config(conn), "head")
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def session(db_engine):
    with db_engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()
