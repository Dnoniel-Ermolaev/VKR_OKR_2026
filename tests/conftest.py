"""Общие фикстуры и ленивая подмена окружения для тестов.

До импорта ``src.infrastructure.db.database`` мы выставляем ``DATABASE_URL``
на SQLite in-memory, чтобы тесты работали автономно и не зависели от Postgres.
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def in_memory_session():
    """SQLite in-memory sqlalchemy session с применёнными миграциями."""
    from src.infrastructure.db.database import Base
    # import models to register them on Base
    from src.infrastructure.db import models  # noqa: F401

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
