from alembic import context
from sqlalchemy import create_engine

from app.core.config import get_settings
from app.storage import models  # noqa: F401  (registers tables on Base.metadata)
from app.storage.database import Base, database_url

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    # Tests may set sqlalchemy.url explicitly; otherwise use the configured data dir.
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured
    data_dir = get_settings().data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return database_url(data_dir)


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
