# src/web/api.py
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.infrastructure.db.database import SessionLocal
from src.web.services import PatientService, TriageService
from src.infrastructure.db.repository import sql_database_repository

app = FastAPI(title="ACS Web API")

# Определяем местоположение файлов HTML и стилей
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

""" FastAPI """
# Автоматически открывает и закрывает БД для каждого запроса
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

""" Маршруты """
# Открываем HTML страницу пользователю
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API: Получение списка пациентов из БД
@app.get("/api/patients")
async def get_patients(db = Depends(get_db)):  # FastAPI сам передаст сюда БД
    service = PatientService(db)
    return service.get_patients_for_sidebar()

# API: Получение расширенного профиля пациента по его ID. 
@app.get("/api/patients/{patient_id}")
async def get_patient_details(patient_id: int, db = Depends(get_db)):
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
                "iso_date": visit.admission_time.isoformat()
            } 
            for visit in patient.visits
        ]
    }

# API: Добавить пациента
@app.post("/api/patients")
async def add_new_patient(data: dict, db = Depends(get_db)):
    service = PatientService(db)
    return service.add_patient(
        last_name = data.get("last_name"),
        first_name = data.get("first_name"),
        patronymic = data.get("patronymic", ""),
        birth_date_str = data.get("birth_date"),
        gender = data.get("gender"))

# API: Обновить пациента
@app.put("/api/patients/{patient_id}")
async def update_patient(patient_id: int, data: dict, db = Depends(get_db)):
    service = PatientService(db)
    return service.update_patient(
        patient_id = patient_id,
        last_name = data.get("last_name"),
        first_name = data.get("first_name"),
        patronymic = data.get("patronymic", ""),
        birth_date_str = data.get("birth_date"),
        gender = data.get("gender")
    )

# API: Удалить пациента
@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: int, db = Depends(get_db)):
    service = PatientService(db)
    return service.delete_patient(patient_id)

# API: Добавить новый визит
@app.post("/api/visits")
async def add_new_visit(data: dict, db = Depends(get_db)):
    patient_id = data.get("patient_id")
    date_str = data.get("date")
    
    service = PatientService(db)
    return service.add_visit(patient_id, date_str)

# API: Удалить визит
@app.delete("/api/visits/{visit_id}")
async def delete_visit(visit_id: int, db = Depends(get_db)):
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
