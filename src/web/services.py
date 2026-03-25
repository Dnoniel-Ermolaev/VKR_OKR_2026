# src/web/services.py
import shlex
from datetime import datetime
from sqlalchemy.orm import Session
from src.cli.main import CLIParser
from src.core.WorkflowRunner import workflow_runner
from src.infrastructure.db.models import Visit, Patient
from src.infrastructure.db.repository import sql_database_repository

class PatientService:

    """
    Сервис для управления бизнес-логикой работы с пациентами.
    Связывает контроллеры API с репозиторием базы данных и выполняет 
    преобразование данных между моделями БД и форматом фронтенда.
    """

    def __init__(self, db_session: Session):
        # :param db_session: Активная сессия SQLAlchemy для работы с базой данных.

        self.repo = sql_database_repository(db_session)
        self.db_session = db_session

    # Получает список всех пациентов для отображения в боковой панели.
    def get_patients_for_sidebar(self) -> list[dict]:
        patients = self.repo.get_all_patients()
        return [
            {
                "id": patient.id,
                "full_name": patient.full_name,
                "birth_date": patient.birth_date.strftime("%d.%m.%Y"),
                "display_id": f"П{patient.id:06d}",
                "risk_color": "#94a3b8" # В дальнейшем, можно устанавливать цвет карточки в зависимости от риска
            }
            for patient in patients
        ]
    
    def add_visit(self, patient_id: int, date_str: str):

        """
        Регистрирует новый визит
        :param patient_id: Внутренний ID пациента.
        :param date_str: Строка даты и времени в формате ISO.
        """

        try:
            visit_date = datetime.fromisoformat(date_str)
            new_visit = Visit(patient_id=patient_id, admission_time=visit_date)
            
            self.db_session.add(new_visit)
            self.db_session.commit()

            return {"success": True, "visit_id": new_visit.id}
        
        except Exception as e:
            self.db_session.rollback() # Стираем все неудачные попытки записи
            return {"error": str(e)}

    def delete_visit(self, visit_id: int):

        """
        Удаляет запись о визите по его идентификатору
        :param visit_id: ID визита в базе данных
        """

        try:
            visit = self.db_session.query(Visit).filter(Visit.id == visit_id).first()
            if not visit:
                return {"error": "Визит не найден"}
            
            self.db_session.delete(visit)
            self.db_session.commit()

            return {"success": True}
        except Exception as e:
            self.db_session.rollback()
            return {"error": str(e)}
        
    def add_patient(self, last_name: str, first_name: str, patronymic: str, birth_date_str: str, gender: str):
            
        """
        Регистрирует нового пациента в системе
        Собирает полное имя из частей и преобразует строку даты в объект date
        """

        try:
            full_name = f"{last_name} {first_name}"
            if patronymic:
                full_name += f" {patronymic}"
                
            b_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
            
            new_patient = Patient(
                full_name=full_name.strip(),
                birth_date=b_date,
                gender=gender
            )
            self.db_session.add(new_patient)
            self.db_session.commit()
            
            # Возвращаем красивый ID
            display_id = f"П{new_patient.id:06d}"
            return {"success": True, "display_id": display_id, "id": new_patient.id}
        except Exception as e:
            self.db_session.rollback()
            return {"error": str(e)}
            
    # Обновляет персональные данные существующего пациента.
    def update_patient(self, patient_id: int, last_name: str, first_name: str, patronymic: str, birth_date_str: str, gender: str):
        try:
            patient = self.db_session.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                return {"error": "Пациент не найден"}

            full_name = f"{last_name} {first_name}"
            if patronymic:
                full_name += f" {patronymic}"
                
            b_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
            
            # Обновляем поля
            patient.full_name = full_name.strip()
            patient.birth_date = b_date
            patient.gender = gender
            
            self.db_session.commit()
            return {"success": True, "message": "Данные пациента обновлены"}
        except Exception as e:
            self.db_session.rollback()
            return {"error": str(e)}

    # удаляет пациента и все связанные с ним данные
    def delete_patient(self, patient_id: int):
        try:
            patient = self.db_session.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                return {"error": "Пациент не найден"}
            
            # Благодаря cascade="all, delete-orphan" в models.py, 
            # все визиты удалятся автоматически вместе с пациентом
            self.db_session.delete(patient)
            self.db_session.commit()
            return {"success": True, "message": "Пациент и все его данные удалены"}
        except Exception as e:
            self.db_session.rollback()
            return {"error": str(e)}

class TriageService:
    """ Сервис для работы с нейросетью и графом """

    @staticmethod
    def process_web_form(data: dict) -> dict:
        app_config = {"require_llm": False, "force_llm": False, "llm_model": "qwen2.5:7b-instruct"}
        return workflow_runner.run_single(data, app_config)

    @staticmethod
    def process_console_command(command: str) -> dict:

        # Убираем все переносы строк и слеши, превращая в одну длинную строку
        clean_cmd = command.replace("\\\n", " ").replace("\\", " ").replace("\n", " ")
        if "src.cli.main" in clean_cmd:
            clean_cmd = clean_cmd.split("src.cli.main")[1].strip()
        
        # Парсим строку
        cli = CLIParser()
        parsed_args = cli.parser.parse_args(shlex.split(clean_cmd))
        
        raw_cli_data = vars(parsed_args)
        app_config = {"require_llm": parsed_args.require_llm,
                      "force_llm": parsed_args.force_llm,
                      "llm_model": parsed_args.model
                      }
        
        return workflow_runner.run_single(raw_cli_data, app_config)

    