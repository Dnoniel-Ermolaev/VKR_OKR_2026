"""Добавление недостающих колонок/таблиц к уже существующей БД.

`Base.metadata.create_all()` не изменяет существующие таблицы — при обновлении кода
старый PostgreSQL остаётся без `case_observations.code` и т.п., что даёт 500 на /control.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def apply_schema_compat(engine: Engine) -> None:
    dialect = engine.dialect.name
    if dialect == "postgresql":
        _upgrade_postgresql(engine)
    elif dialect == "sqlite":
        _upgrade_sqlite(engine)
    # другие СУБД — только create_all снаружи


def _upgrade_postgresql(engine: Engine) -> None:
    stmts = [
        # case_observations — доводим старую таблицу до текущей модели CaseObservation
        """
        ALTER TABLE case_observations
        ADD COLUMN IF NOT EXISTS code VARCHAR(48) NOT NULL DEFAULT '';
        """,
        """
        ALTER TABLE case_observations
        ADD COLUMN IF NOT EXISTS flag VARCHAR(16) NOT NULL DEFAULT 'unknown';
        """,
        """
        ALTER TABLE case_observations
        ADD COLUMN IF NOT EXISTS value_text TEXT NULL;
        """,
        """
        ALTER TABLE case_observations
        ADD COLUMN IF NOT EXISTS source VARCHAR(64) NOT NULL DEFAULT 'manual';
        """,
        """
        ALTER TABLE case_observations
        ADD COLUMN IF NOT EXISTS note TEXT NOT NULL DEFAULT '';
        """,
        """
        ALTER TABLE triage_cases
        ADD COLUMN IF NOT EXISTS closed_at TIMESTAMP WITH TIME ZONE NULL;
        """,
    ]
    with engine.begin() as conn:
        for sql in stmts:
            conn.execute(text(sql))
        # длина name для длинных названий из каталога
        try:
            conn.execute(
                text("ALTER TABLE case_observations ALTER COLUMN name TYPE VARCHAR(120);")
            )
        except Exception:
            pass


def _upgrade_sqlite(engine: Engine) -> None:
    """SQLite до 3.35 без IF NOT EXISTS для колонок — проверяем pragma_table_info."""
    with engine.connect() as raw:
        exists = raw.execute(
            text(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='case_observations' LIMIT 1;"
            )
        ).fetchone()
        if not exists:
            return
        rows = raw.execute(text("PRAGMA table_info(case_observations);")).fetchall()
        colnames = {r[1] for r in rows}
    alters = []
    if "code" not in colnames:
        alters.append("ALTER TABLE case_observations ADD COLUMN code VARCHAR(48) NOT NULL DEFAULT '';")
    if "flag" not in colnames:
        alters.append("ALTER TABLE case_observations ADD COLUMN flag VARCHAR(16) NOT NULL DEFAULT 'unknown';")
    if "value_text" not in colnames:
        alters.append("ALTER TABLE case_observations ADD COLUMN value_text TEXT NULL;")
    if "source" not in colnames:
        alters.append("ALTER TABLE case_observations ADD COLUMN source VARCHAR(64) NOT NULL DEFAULT 'manual';")
    if "note" not in colnames:
        alters.append("ALTER TABLE case_observations ADD COLUMN note TEXT NOT NULL DEFAULT '';")
    if alters:
        with engine.begin() as conn:
            for sql in alters:
                conn.execute(text(sql))
    with engine.connect() as raw:
        if not raw.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='triage_cases' LIMIT 1;")
        ).fetchone():
            return
        rows = raw.execute(text("PRAGMA table_info(triage_cases);")).fetchall()
        tc_cols = {r[1] for r in rows}
    if "closed_at" not in tc_cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE triage_cases ADD COLUMN closed_at TIMESTAMP NULL;"))
