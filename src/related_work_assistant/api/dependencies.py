from collections.abc import Iterator

from sqlalchemy import Engine

from related_work_assistant.core.config import Settings
from related_work_assistant.database.session import make_engine


def get_engine() -> Iterator[Engine]:
    engine = make_engine(Settings())
    try:
        yield engine
    finally:
        engine.dispose()
