"""Однократное обновление схемы БД под текущие модели (ALTER + create_all).

Запуск из корня репозитория:
    python -m scripts.upgrade_db_schema

Нужно, если после `git pull` в логах PostgreSQL: «столбец case_observations.code не существует».
"""
from __future__ import annotations

from src.infrastructure.db import models  # noqa: F401
from src.infrastructure.db.database import Base, engine
from src.infrastructure.db.schema_upgrade import apply_schema_compat


def main() -> None:
    print("create_all (новые таблицы) …")
    Base.metadata.create_all(bind=engine)
    print("apply_schema_compat (колонки в старых таблицах) …")
    apply_schema_compat(engine)
    print("Готово.")


if __name__ == "__main__":
    main()
