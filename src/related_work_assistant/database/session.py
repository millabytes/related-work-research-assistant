from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from related_work_assistant.core.config import Settings


def make_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.sqlalchemy_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        hide_parameters=True,
        connect_args={"connect_timeout": 3},
    )


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    with sessionmaker(engine, expire_on_commit=False).begin() as session:
        yield session
