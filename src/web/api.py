# src/web/api.py
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.infrastructure.db.database import SessionLocal
from src.infrastructure.db.init_db import init_database
from src.infrastructure.db.repository import sql_database_repository
from src.infrastructure.importers.excel_importer import ExcelImportService
from src.web.services import (
    CaseService,
    CatalogService,
    DiagnosisService,
    GraphTraceService,
    MedicationService,
    ObservationService,
    PatientService,
    ProcedureService,
    ReassessService,
    StudyService,
    TriageService,
)

app = FastAPI(title="ACS Web API")


# Подтягиваем схему БД при старте (идемпотентно). Это спасает от ситуации,
# когда после обновления кода в репозитории появились новые колонки
# (например, latest_acs_diagnosis / latest_path_trace_json), а разработчик
# забыл выполнить `python -m src.infrastructure.db.init_db` вручную.
@app.on_event("startup")
def _ensure_schema_up_to_date() -> None:
    try:
        init_database()
    except Exception as exc:
        # Если БД недоступна - поднимать сервис всё равно не будем,
        # но падать целиком тоже не хотим: пусть FastAPI стартует
        # с понятной ошибкой в логе.
        print(f"[startup] init_database failed: {exc}")


# Определяем местоположение файлов HTML и стилей
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

""" FastAPI """
# Автоматически открывает и закрывает БД для каждого запроса
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


""" Маршруты """
# Открываем HTML страницу пользователю
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# API: Получение списка пациентов из БД
@app.get("/api/patients")
async def get_patients(db=Depends(get_db)):  # FastAPI сам передаст сюда БД
    service = PatientService(db)
    return service.get_patients_for_sidebar()

# API: Получение расширенного профиля пациента по его ID.
@app.get("/api/patients/{patient_id}")
async def get_patient_details(patient_id: int, db=Depends(get_db)):
    repository = sql_database_repository(db)
    patient = repository.get_patient_full_details(patient_id)
    if not patient:
        return {"error": "Пациент не найден"}

    # Собираем подробный ответ
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
                "created_at": case.created_at.isoformat(),
                "updated_at": case.updated_at.isoformat(),
                "closed_at": case.closed_at.isoformat() if case.closed_at else None,
            }
            for case in sorted(patient.cases, key=lambda item: item.created_at, reverse=True)
        ],
    }

# API: Добавить пациента
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

# API: Обновить пациента
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

# API: Удалить пациента
@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: int, db=Depends(get_db)):
    service = PatientService(db)
    return service.delete_patient(patient_id)

# API: Добавить новый визит
@app.post("/api/visits")
async def add_new_visit(data: dict, db=Depends(get_db)):
    patient_id = data.get("patient_id")
    date_str = data.get("date")

    service = PatientService(db)
    return service.add_visit(patient_id, date_str)

# API: Удалить визит
@app.delete("/api/visits/{visit_id}")
async def delete_visit(visit_id: int, db=Depends(get_db)):
    service = PatientService(db)
    return service.delete_visit(visit_id)


# API: Получение данных с Веб-формы. Запуск алгоритма работы с графом
@app.post("/api/assess")
async def api_assess(data: dict):
    try:
        return TriageService.process_web_form(data)
    except Exception as e:
        return {"error": str(e)}

# API: Получение сырой команды из Веб-Консоли
@app.post("/api/console")
async def api_console(payload: dict):
    try:
        raw_cmd = payload.get("command", "")
        return TriageService.process_console_command(raw_cmd)
    except SystemExit:
        return {"error": "Неверный синтаксис команды."}
    except Exception as e:
        return {"error": str(e)}


# API: Получение медицинского каталога для выпадающих списков
@app.get("/api/catalog")
async def api_catalog():
    return CatalogService.payload()


# API: Получение активного стационарного кейса пациента
@app.get("/api/cases/active")
async def api_active_case(patient_id: int | None = None, db=Depends(get_db)):
    service = CaseService(db)
    return service.get_active(patient_id)


# API: Создать новый (стационарный) кейс
@app.post("/api/cases/start")
async def api_start_case(data: dict, db=Depends(get_db)):
    # В новой структуре кейс создается не только по названию, но и по клиническому payload.
    try:
        service = CaseService(db)
        return service.start_case(data)
    except Exception as e:
        return {"error": str(e)}


# API: Получить полную карточку стационарного кейса
@app.get("/api/cases/{case_id}")
async def api_get_case(case_id: str, db=Depends(get_db)):
    try:
        service = CaseService(db)
        return service.get_case(case_id)
    except Exception as e:
        return {"error": str(e)}


# API: Продолжить кейс новыми данными наблюдения
@app.post("/api/cases/{case_id}/resume")
async def api_resume_case(case_id: str, data: dict, db=Depends(get_db)):
    try:
        service = CaseService(db)
        return service.resume_case(case_id, data)
    except Exception as e:
        return {"error": str(e)}


# API: Закрыть кейс
@app.post("/api/cases/{case_id}/close")
async def api_close_case(case_id: str, db=Depends(get_db)):
    service = CaseService(db)
    return service.close_case(case_id)


# API: Переоткрыть закрытый кейс
@app.post("/api/cases/{case_id}/reopen")
async def api_reopen_case(case_id: str, db=Depends(get_db)):
    service = CaseService(db)
    return service.reopen_case(case_id)


# API: Удалить кейс
@app.delete("/api/cases/{case_id}")
async def api_delete_case(case_id: str, db=Depends(get_db)):
    service = CaseService(db)
    return service.delete_case(case_id)


# API: Сгенерировать эпикриз или отчет по кейсу
@app.post("/api/cases/{case_id}/report")
async def api_case_report(case_id: str, data: dict, db=Depends(get_db)):
    try:
        service = CaseService(db)
        return service.generate_report(case_id, data)
    except Exception as e:
        return {"error": str(e)}


# API: Переоценить риск по текущим данным кейса
@app.post("/api/cases/{case_id}/reassess")
async def api_case_reassess(case_id: str, data: dict | None = None, db=Depends(get_db)):
    service = ReassessService(db)
    return service.run(case_id, data or {})


# API: Получить контрольную панель кейса (протокол, готовность, алерты)
@app.get("/api/cases/{case_id}/control")
async def api_case_control(case_id: str, db=Depends(get_db)):
    service = CaseService(db)
    return service.get_control_dashboard(case_id)


# API: Получить канонический граф пациента + трассу прохождения по нему
@app.get("/api/cases/{case_id}/graph-trace")
async def api_case_graph_trace(
    case_id: str,
    assessment_id: int | None = None,
    db=Depends(get_db),
):
    service = GraphTraceService(db)
    return service.get(case_id, assessment_id=assessment_id)


# API: Получить витальные показатели кейса
@app.get("/api/cases/{case_id}/vitals")
async def api_list_vitals(case_id: str, db=Depends(get_db)):
    return ObservationService(db).list_vitals(case_id)


# API: Добавить витальный показатель
@app.post("/api/cases/{case_id}/vitals")
async def api_add_vital(case_id: str, data: dict, db=Depends(get_db)):
    result = ObservationService(db).add_vital(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


# API: Обновить витальный показатель
@app.put("/api/vitals/{observation_id}")
async def api_update_vital(observation_id: int, data: dict, db=Depends(get_db)):
    return ObservationService(db).update(observation_id, data)


# API: Удалить витальный показатель
@app.delete("/api/vitals/{observation_id}")
async def api_delete_vital(observation_id: int, db=Depends(get_db)):
    return ObservationService(db).delete(observation_id)


# API: Получить лабораторные анализы кейса
@app.get("/api/cases/{case_id}/labs")
async def api_list_labs(case_id: str, db=Depends(get_db)):
    return ObservationService(db).list_labs(case_id)


# API: Добавить лабораторный анализ
@app.post("/api/cases/{case_id}/labs")
async def api_add_lab(case_id: str, data: dict, db=Depends(get_db)):
    result = ObservationService(db).add_lab(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


# API: Обновить лабораторный анализ
@app.put("/api/labs/{observation_id}")
async def api_update_lab(observation_id: int, data: dict, db=Depends(get_db)):
    return ObservationService(db).update(observation_id, data)


# API: Удалить лабораторный анализ
@app.delete("/api/labs/{observation_id}")
async def api_delete_lab(observation_id: int, db=Depends(get_db)):
    return ObservationService(db).delete(observation_id)


# API: Получить инструментальные исследования кейса
@app.get("/api/cases/{case_id}/studies")
async def api_list_studies(case_id: str, db=Depends(get_db)):
    return StudyService(db).list(case_id)


# API: Добавить инструментальное исследование
@app.post("/api/cases/{case_id}/studies")
async def api_add_study(case_id: str, data: dict, db=Depends(get_db)):
    result = StudyService(db).add(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


# API: Обновить инструментальное исследование
@app.put("/api/studies/{item_id}")
async def api_update_study(item_id: int, data: dict, db=Depends(get_db)):
    return StudyService(db).update(item_id, data)


# API: Удалить инструментальное исследование
@app.delete("/api/studies/{item_id}")
async def api_delete_study(item_id: int, db=Depends(get_db)):
    return StudyService(db).delete(item_id)


# API: Получить процедуры и вмешательства кейса
@app.get("/api/cases/{case_id}/procedures")
async def api_list_procedures(case_id: str, db=Depends(get_db)):
    return ProcedureService(db).list(case_id)


# API: Добавить процедуру или вмешательство
@app.post("/api/cases/{case_id}/procedures")
async def api_add_procedure(case_id: str, data: dict, db=Depends(get_db)):
    result = ProcedureService(db).add(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


# API: Обновить процедуру или вмешательство
@app.put("/api/procedures/{item_id}")
async def api_update_procedure(item_id: int, data: dict, db=Depends(get_db)):
    return ProcedureService(db).update(item_id, data)


# API: Удалить процедуру или вмешательство
@app.delete("/api/procedures/{item_id}")
async def api_delete_procedure(item_id: int, db=Depends(get_db)):
    return ProcedureService(db).delete(item_id)


# API: Получить назначения по кейсу
@app.get("/api/cases/{case_id}/medications")
async def api_list_medications(case_id: str, db=Depends(get_db)):
    return MedicationService(db).list(case_id)


# API: Добавить назначение
@app.post("/api/cases/{case_id}/medications")
async def api_add_medication(case_id: str, data: dict, db=Depends(get_db)):
    result = MedicationService(db).add(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


# API: Обновить назначение
@app.put("/api/medications/{item_id}")
async def api_update_medication(item_id: int, data: dict, db=Depends(get_db)):
    return MedicationService(db).update(item_id, data)


# API: Удалить назначение
@app.delete("/api/medications/{item_id}")
async def api_delete_medication(item_id: int, db=Depends(get_db)):
    return MedicationService(db).delete(item_id)


# API: Получить диагнозы кейса
@app.get("/api/cases/{case_id}/diagnoses")
async def api_list_diagnoses(case_id: str, db=Depends(get_db)):
    return DiagnosisService(db).list(case_id)


# API: Добавить диагноз
@app.post("/api/cases/{case_id}/diagnoses")
async def api_add_diagnosis(case_id: str, data: dict, db=Depends(get_db)):
    result = DiagnosisService(db).add(case_id, data)
    if _should_reassess(data) and "error" not in result:
        ReassessService(db).run(case_id, data)
    return result


# API: Обновить диагноз
@app.put("/api/diagnoses/{item_id}")
async def api_update_diagnosis(item_id: int, data: dict, db=Depends(get_db)):
    return DiagnosisService(db).update(item_id, data)


# API: Удалить диагноз
@app.delete("/api/diagnoses/{item_id}")
async def api_delete_diagnosis(item_id: int, db=Depends(get_db)):
    return DiagnosisService(db).delete(item_id)


# API: Скачать Excel-шаблон для одного листа
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


# API: Скачать полный Excel-шаблон для импорта данных кейса
@app.get("/api/cases/{case_id}/excel-template")
async def api_excel_template_full(case_id: str):
    importer = ExcelImportService()
    content = importer.build_template_full()
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="acs_template_full.xlsx"'},
    )


# API: Загрузить Excel-файл с данными кейса
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


# Вспомогательная проверка: нужно ли автоматически запускать переоценку после записи данных
def _should_reassess(data: dict | None) -> bool:
    if not data:
        return False
    return bool(data.get("auto_reassess", False))
