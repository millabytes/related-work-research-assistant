"""Alembic owns schema creation; application configuration owns credentials."""

from alembic import context

from related_work_assistant.core.config import Settings
from related_work_assistant.database import models  # noqa: F401
from related_work_assistant.database.base import Base
from related_work_assistant.database.session import make_engine

config = context.config
target_metadata = Base.metadata


def run_migrations() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    elif context.is_offline_mode():
        context.configure(
            url=Settings().sqlalchemy_url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
        )
        with context.begin_transaction():
            context.run_migrations()
    else:
        engine = make_engine(Settings())
        try:
            with engine.connect() as connection:
                context.configure(connection=connection, target_metadata=target_metadata)
                with context.begin_transaction():
                    context.run_migrations()
        finally:
            engine.dispose()


run_migrations()
