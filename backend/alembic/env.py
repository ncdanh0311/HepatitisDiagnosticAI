"""
backend/alembic/env.py
=======================
Alembic migration environment.
- Offline mode: generates pure SQL from metadata (no live DB needed).
- Online mode: connects async via asyncpg for `alembic upgrade head`.
"""
import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import create_async_engine

# Import all ORM models so Alembic detects them
from app.db.base import Base  # noqa: F401
import app.models.user        # noqa: F401
import app.models.patient     # noqa: F401
import app.models.prediction  # noqa: F401
import app.models.chat_log    # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
    # Alembic needs a sync URL for offline / comparison — strip +asyncpg
    return url.replace("postgresql+asyncpg", "postgresql")


def run_migrations_offline() -> None:
    """Generate SQL script without connecting to the database."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Apply migrations using async asyncpg connection."""
    url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
    if not url.startswith("postgresql+asyncpg"):
        url = url.replace("postgresql", "postgresql+asyncpg", 1)
    connectable = create_async_engine(url, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(
            lambda conn: context.configure(
                connection=conn,
                target_metadata=target_metadata,
                compare_type=True,
            )
        )
        await connection.run_sync(lambda _: context.run_migrations())
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
