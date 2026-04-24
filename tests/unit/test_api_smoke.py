"""Smoke-тесты FastAPI: каталог, шаблоны Excel, публичные endpoint'ы."""
from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
openpyxl = pytest.importorskip("openpyxl")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def client():
    """FastAPI TestClient с подменой `get_db` на SQLite in-memory."""
    from src.infrastructure.db.database import Base
    from src.infrastructure.db import models  # noqa: F401  (ensure models registered)
    from src.web.api import app, get_db

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def _override():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_home_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_catalog_endpoint_returns_all_sections(client):
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    payload = resp.json()
    for key in ("vitals", "labs", "studies", "procedures", "medications", "diagnoses"):
        assert key in payload
        assert isinstance(payload[key], list)
        assert payload[key]


def test_patients_list_empty_by_default(client):
    resp = client.get("/api/patients")
    assert resp.status_code == 200
    assert resp.json() == []


def test_active_case_returns_none_without_data(client):
    resp = client.get("/api/cases/active")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("case") is None


def test_excel_template_full_is_valid_xlsx(client):
    resp = client.get("/api/cases/any-id/excel-template")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    assert "Labs" in wb.sheetnames
    assert "Vitals" in wb.sheetnames


def test_excel_template_single_sheet(client):
    resp = client.get("/api/cases/any-id/excel-template/Labs")
    assert resp.status_code == 200
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    assert "Labs" in wb.sheetnames


def test_excel_template_unknown_sheet_returns_404(client):
    resp = client.get("/api/cases/any-id/excel-template/NotASheet")
    assert resp.status_code == 404


def test_patient_crud_flow(client):
    created = client.post(
        "/api/patients",
        json={
            "last_name": "Иванов",
            "first_name": "Иван",
            "patronymic": "Иванович",
            "birth_date": "1960-01-15",
            "gender": "male",
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body.get("success") or body.get("id") or "error" not in body

    listing = client.get("/api/patients")
    assert listing.status_code == 200
    patients = listing.json()
    assert isinstance(patients, list)
    assert len(patients) >= 1
