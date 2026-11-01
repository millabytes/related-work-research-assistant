from sqlalchemy import Engine, text


def check_database(engine: Engine) -> bool:
    with engine.connect() as connection:
        return connection.scalar(text("SELECT 1")) == 1 and bool(
            connection.scalar(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
        )
