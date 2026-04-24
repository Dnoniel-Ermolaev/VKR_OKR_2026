# src/web/api.py
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.infrastructure.db.database import SessionLocal
from src.infrastructure.db.repository import sql_database_repository
from src.infrastructure.importers.excel_importer import ExcelImportService
from src.web.services import (
    CaseService,
    CatalogService,
    DiagnosisService,
    MedicationService,
    ObservationService,
    PatientService,
    ProcedureService,
    ReassessService,
    StudyService,
    TriageService,
)

app = FastAPI(title="ACS Web API")

app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")


@app.on_event("startup")
def _ensure_database_schema() -> None:
    """Старые БД без новых колонок ломают /control; create_all не делает ALTER."""
    from src.infrastructure.db import models  # noqa: F401
    from src.infrastructure.db.database import Base, engine
    from src.infrastructure.db.schema_upgrade import apply_schema_compat

    Base.metadata.create_all(bind=engine)
    apply_schema_compat(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------- Pages ----------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ---------------------------- Patients / visits ----------------------------
@app.get("/api/patients")
async def get_patients(db=Depends(get_db)):
    service = PatientService(db)
    return service.get_patients_for_sidebar()


@app.get("/api/patients/{patient_id}")
async def get_patient_details(patient_id: int, db=Depends(get_db)):
    repository = sql_database_repository(db)
    patient = repository.get_patient_full_details(patient_id)
    if not patient:
        return {"error": "Пациент не найден"}
    return {
        "id": patient.id,
        "display_id": f"П{patient.id:06d}",
        "full_name": patient.full_name,
        "birth_date": patient.birth_date.strftime("%d.%m.%Y"),
        "gender": patient.gender,
        "visits": [
            {
                "id": visit.id,
                "date": visit.admission_time.strftime("%d.%m.%Y %H:%M"),
                "iso_date": visit.admission_time.isoformat(),
            }
            for visit in patient.visits
        ],
        "cases": [
            {
                "id": case.id,
                "title": case.title,
                "status": case.status,
                "current_stage": case.current_stage,
                "latest_risk_level": case.latest_risk_level,
                "latest_triage_category": case.latest_triage_category,
                "visit_id": case.visit_id,
                "created_at": case.created_at.isoformat(),
                "updated_at": case.updated_at.isoformat(),
                "closed_at": case.closed_at.isoformat() if case.closed_at else None,
            }
            for case in sorted(patient.cases, key=lambda item: item.created_at, reverse=True)
        ],
    }


@app.post("/api/patients")
async def add_new_patient(data: dict, db=Depends(get_db)):
    service = PatientService(db)
    return service.add_patient(
        last_name=data.get("last_name"),
        first_name=data.get("first_name"),
        patronymic=data.get("patronymic", ""),
        birth_date_str=data.get("birth_date"),
        gender=data.get("gender"),
    )


@app.put("/api/patients/{patient_id}")
async def update_patient(patient_id: int, data: dict, db=Depends(get_db)):
    service = PatientService(db)
    return service.update_patient(
        patient_id=patient_id,
        last_name=data.get("last_name"),
        first_name=data.get("first_name"),
        patronymic=data.get("patronymic", ""),
        birth_date_str=data.get("birth_date"),
        gender=data.get("gender"),
    )


@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: int, db=Depends(get_db)):
    service = PatientService(db)
    return service.delete_patient(patient_id)


@app.post("/api/visits")
async def add_new_visit(data: dict, db=Depends(get_db)):
    service = PatientService(db)
    return service.add_visit(data.get("patient_id"), data.get("date"))


@app.delete("/api/visits/{visit_id}")
async def delete_visit(visit_id: int, db=Depends(get_db)):
    service = PatientService(db)
    return service.delete_visit(visit_id)


# ---------------------------- Triage / console ----------------------------
@app.post("/api/assess")
async def api_assess(data: dict):
    try:
        return TriageService.process_web_form(data)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/console")
async def api_console(payload: dict):
    try:
        raw_cmd = payload.get("command", "")
        return TriageService.process_console_command(raw_cmd)
    except SystemExit:
        return {"error": "Неверный синтаксис команды."}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------- Catalog ----------------------------
@app.get("/api/catalog")
async def api_catalog():
    return CatalogService.payload()


# ---------------------------- Case lifecycle ----------------------------
@app.get("/api/cases/active")
async def api_active_case(patient_id: int | None = None, visit_id: int | None = None, db=Depends(get_db)):
    service = CaseService(db)
    return service.get_active(patient_id, visit_id)


@app.post("/api/cases/start")
async def api_start_case(data: dict, db=Depends(get_db)):
    try:
        service = CaseService(db)
        return service.start_case(data)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/cases/{case_id}")
async def api_get_case(case_id: str, db=Depends(get_db)):
    try:
        service = CaseService(db)
        return service.get_case(case_id)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/cases/{case_id}/resume")
async def api_resume_case(case_id: str, data: dict, db=Depends(get_db)):
    try:
        service = CaseService(db)
        return service.resume_case(case_id, data)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/cases/{case_id}/close")
async def api_close_case(case_id: str, db=Depends(get_db)):
    service = CaseService(db)
    return service.close_case(case_id)


@app.post("/api/cases/{case_id}/reopen")
async def api_reopen_case(case_id: str, db=Depends(get_db)):
    service = CaseService(db)
    return service.reopen_case(case_id)


@app.post("/api/cases/{case_id}/report")
async def api_case_report(case_id: str, data: dict, db=Depends(get_db)):
    try:
        service = CaseService(db)
        return service.generate_report(case_id, data)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/cases/{case_id}/reassess")
async def api_case_reassess(case_id: str, data: dict | None = None, db=Depends(get_db)):
    service = ReassessService(db)
    return service.run(case_id, data or {})


@app.get("/api/cases/{case_id}/control")
async def api_case_control(case_id: str, db=Depends(get_db)):
    service = CaseService(db)
    return service.get_control_dashboard(case_id)


# ---------------------------- Vitals ----------------------------
@app.get("/api/cases/{case_id}/vitals")
async def api_list_vitals(case_id: str, db=Depends(get_db)):
    return ObservationService(db).list_vitals(case_id)


@app.post("/api/cases/{case_id}/vitals")
async def api_add_vital(case_id: str, data: dict, db=Depends(get_db)):
    result = ObservationService(db).add_vital(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


@app.put("/api/vitals/{observation_id}")
async def api_update_vital(observation_id: int, data: dict, db=Depends(get_db)):
    return ObservationService(db).update(observation_id, data)


@app.delete("/api/vitals/{observation_id}")
async def api_delete_vital(observation_id: int, db=Depends(get_db)):
    return ObservationService(db).delete(observation_id)


# ---------------------------- Labs ----------------------------
@app.get("/api/cases/{case_id}/labs")
async def api_list_labs(case_id: str, db=Depends(get_db)):
    return ObservationService(db).list_labs(case_id)


@app.post("/api/cases/{case_id}/labs")
async def api_add_lab(case_id: str, data: dict, db=Depends(get_db)):
    result = ObservationService(db).add_lab(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


@app.put("/api/labs/{observation_id}")
async def api_update_lab(observation_id: int, data: dict, db=Depends(get_db)):
    return ObservationService(db).update(observation_id, data)


@app.delete("/api/labs/{observation_id}")
async def api_delete_lab(observation_id: int, db=Depends(get_db)):
    return ObservationService(db).delete(observation_id)


# ---------------------------- Studies ----------------------------
@app.get("/api/cases/{case_id}/studies")
async def api_list_studies(case_id: str, db=Depends(get_db)):
    return StudyService(db).list(case_id)


@app.post("/api/cases/{case_id}/studies")
async def api_add_study(case_id: str, data: dict, db=Depends(get_db)):
    result = StudyService(db).add(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


@app.put("/api/studies/{item_id}")
async def api_update_study(item_id: int, data: dict, db=Depends(get_db)):
    return StudyService(db).update(item_id, data)


@app.delete("/api/studies/{item_id}")
async def api_delete_study(item_id: int, db=Depends(get_db)):
    return StudyService(db).delete(item_id)


# ---------------------------- Procedures ----------------------------
@app.get("/api/cases/{case_id}/procedures")
async def api_list_procedures(case_id: str, db=Depends(get_db)):
    return ProcedureService(db).list(case_id)


@app.post("/api/cases/{case_id}/procedures")
async def api_add_procedure(case_id: str, data: dict, db=Depends(get_db)):
    result = ProcedureService(db).add(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


@app.put("/api/procedures/{item_id}")
async def api_update_procedure(item_id: int, data: dict, db=Depends(get_db)):
    return ProcedureService(db).update(item_id, data)


@app.delete("/api/procedures/{item_id}")
async def api_delete_procedure(item_id: int, db=Depends(get_db)):
    return ProcedureService(db).delete(item_id)


# ---------------------------- Medications ----------------------------
@app.get("/api/cases/{case_id}/medications")
async def api_list_medications(case_id: str, db=Depends(get_db)):
    return MedicationService(db).list(case_id)


@app.post("/api/cases/{case_id}/medications")
async def api_add_medication(case_id: str, data: dict, db=Depends(get_db)):
    result = MedicationService(db).add(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


@app.put("/api/medications/{item_id}")
async def api_update_medication(item_id: int, data: dict, db=Depends(get_db)):
    return MedicationService(db).update(item_id, data)


@app.delete("/api/medications/{item_id}")
async def api_delete_medication(item_id: int, db=Depends(get_db)):
    return MedicationService(db).delete(item_id)


# ---------------------------- Diagnoses ----------------------------
@app.get("/api/cases/{case_id}/diagnoses")
async def api_list_diagnoses(case_id: str, db=Depends(get_db)):
    return DiagnosisService(db).list(case_id)


@app.post("/api/cases/{case_id}/diagnoses")
async def api_add_diagnosis(case_id: str, data: dict, db=Depends(get_db)):
    result = DiagnosisService(db).add(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


@app.put("/api/diagnoses/{item_id}")
async def api_update_diagnosis(item_id: int, data: dict, db=Depends(get_db)):
    return DiagnosisService(db).update(item_id, data)


@app.delete("/api/diagnoses/{item_id}")
async def api_delete_diagnosis(item_id: int, db=Depends(get_db)):
    return DiagnosisService(db).delete(item_id)


# ---------------------------- Excel ----------------------------
@app.get("/api/cases/{case_id}/excel-template/{sheet}")
async def api_excel_template(case_id: str, sheet: str):
    importer = ExcelImportService()
    try:
        content = importer.build_template(sheet)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    filename = f"acs_template_{sheet}.xlsx"
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/cases/{case_id}/excel-template")
async def api_excel_template_full(case_id: str):
    importer = ExcelImportService()
    content = importer.build_template_full()
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="acs_template_full.xlsx"'},
    )


@app.post("/api/cases/{case_id}/excel-import")
async def api_excel_import(
    case_id: str,
    file: UploadFile = File(...),
    dry_run: bool = False,
    db=Depends(get_db),
):
    content = await file.read()
    importer = ExcelImportService(db_session=db)
    result = importer.import_case_data(case_id, content, dry_run=dry_run)
    if not dry_run and result.get("imported_total", 0) > 0:
        try:
            ReassessService(db).run(case_id, {})
        except Exception as e:
            result["reassess_error"] = str(e)
    return result


def _should_reassess(data: dict | None) -> bool:
    if not data:
        return False
    return bool(data.get("auto_reassess", False))
