from sqlalchemy import text

from src.infrastructure.db.database import engine, Base
from src.infrastructure.db.models import (  # noqa: F401 - ensure models are registered with Base
    CaseAssessment,
    CaseDiagnosis,
    CaseMedication,
    CaseObservation,
    CaseProcedure,
    CaseStudy,
    CaseTrackingItem,
    ClinicalReport,
    Patient,
    TriageCase,
    Visit,
)

""" Скрипт начальной инициализации и безопасного обновления схемы базы """

# Запуск из корня проекта:
# python -m src.infrastructure.db.init_db

SCHEMA_UPGRADE_SQL = [
    # Колонки трассы по графу и диагностической метки добавлены в Track 4.
    # Для существующих PostgreSQL-инсталляций добавляем без удаления данных.
    "ALTER TABLE case_assessments ADD COLUMN IF NOT EXISTS path_trace_json JSON DEFAULT '{}'::json;",
    "ALTER TABLE case_assessments ADD COLUMN IF NOT EXISTS acs_diagnosis_json JSON DEFAULT '{}'::json;",
    "ALTER TABLE triage_cases ADD COLUMN IF NOT EXISTS latest_path_trace_json JSON DEFAULT '{}'::json;",
    "ALTER TABLE triage_cases ADD COLUMN IF NOT EXISTS latest_acs_diagnosis JSON DEFAULT '{}'::json;",
]


def init_database() -> None:
    """ Создание таблиц и выполнение ручных SQL-обновлений схемы """
    print("Создание недостающих таблиц в базе ...")
    _create_missing_tables()
    print("Выполнение ручных обновлений схемы ...")
    _apply_schema_upgrades()
    print("Инициализация базы завершена.")


def _create_missing_tables() -> None:
    """ Создание таблиц в одной транзакции """
    with engine.begin() as connection:
        Base.metadata.create_all(bind=connection)


def _apply_schema_upgrades() -> None:
    """ Выполнение SQL-команд для ручного добавления колонок """
    for sql in SCHEMA_UPGRADE_SQL:
        try:
            with engine.begin() as connection:
                connection.execute(text(sql))
            print(f"Выполнено обновление схемы: {sql}")
        except Exception as exc:
            print(f"Не удалось выполнить обновление схемы: {sql}. Ошибка: {exc}")

if __name__ == "__main__":
    init_database()
