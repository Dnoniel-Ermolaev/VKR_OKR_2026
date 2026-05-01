# src/web/services.py
import shlex
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.cli.main import CLIParser
from src.core.WorkflowRunner import workflow_runner
from src.core.case_runtime import build_case_title, build_series, normalize_observations
from src.core.patient_control import build_case_control
from src.infrastructure.db.models import Patient, TriageCase, Visit
from src.infrastructure.db.repository import sql_database_repository
from src.medical.catalog import (
    DIAGNOSIS_BY_ICD,
    LAB_BY_CODE,
    MEDICATION_BY_CODE,
    PROCEDURE_BY_CODE,
    STUDY_BY_CODE,
    VITAL_BY_CODE,
    catalog_as_json,
    flag_for_lab,
    flag_for_vital,
)
from src.medical.protocols import protocol_summary


class PatientService:
    """
    Сервис для управления бизнес-логикой работы с пациентами.
    Связывает контроллеры API с базой данных и преобразует данные пациента
    в формат, удобный для отображения на веб-странице.
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
                "risk_color": "#94a3b8",
            }
            for patient in patients
        ]

    def add_visit(self, patient_id: int, date_str: str):
        """
        Регистрирует новый визит пациента
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
        :param visit_id: ID визита в базе данных.
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
            new_patient = Patient(full_name=full_name.strip(), birth_date=b_date, gender=gender)
            self.db_session.add(new_patient)
            self.db_session.commit()

            # Возвращаем красивый ID для отображения в интерфейсе
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

            # Обновляем поля карточки пациента
            patient.full_name = full_name.strip()
            patient.birth_date = b_date
            patient.gender = gender
            self.db_session.commit()
            return {"success": True, "message": "Данные пациента обновлены"}
        except Exception as e:
            self.db_session.rollback()
            return {"error": str(e)}

    # Удаляет пациента и все связанные с ним визиты, кейсы и медицинские данные.
    def delete_patient(self, patient_id: int):
        try:
            patient = self.db_session.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                return {"error": "Пациент не найден"}
            # Явный порядок: кейсы (со всеми дочерними сущностями по cascade ORM) → визиты → пациент.
            cases = self.db_session.query(TriageCase).filter(TriageCase.patient_id == patient_id).all()
            for case in cases:
                self.db_session.delete(case)
            visits = self.db_session.query(Visit).filter(Visit.patient_id == patient_id).all()
            for visit in visits:
                self.db_session.delete(visit)
            self.db_session.delete(patient)
            self.db_session.commit()
            return {"success": True, "message": "Пациент и все его данные удалены"}
        except Exception as e:
            self.db_session.rollback()
            return {"error": str(e)}


class TriageService:
    """
    Сервис для работы с нейросетью и графом.
    Принимает данные из веб-формы или консоли и передает их в WorkflowRunner.
    """

    @staticmethod
    def process_web_form(data: dict) -> dict:
        """
        Запускает оценку риска по данным, введенным в веб-форме
        Использует быстрый режим без обязательного вызова LLM.
        """
        app_config = {"require_llm": False, "force_llm": False, "llm_model": "qwen2.5:7b-instruct"}
        return workflow_runner.run_single(data, app_config)

    @staticmethod
    def process_console_command(command: str) -> dict:
        """
        Обрабатывает команду из веб-консоли
        Преобразует многострочный ввод в строку CLI и запускает общий граф.
        """
        # Убираем все переносы строк и слеши, превращая ввод в одну длинную строку
        clean_cmd = command.replace("\\\n", " ").replace("\\", " ").replace("\n", " ")
        if "src.cli.main" in clean_cmd:
            clean_cmd = clean_cmd.split("src.cli.main")[1].strip()

        # Парсим строку так же, как обычную команду CLI
        cli = CLIParser()
        parsed_args = cli.parser.parse_args(shlex.split(clean_cmd))
        raw_cli_data = vars(parsed_args)
        app_config = {
            "require_llm": parsed_args.require_llm,
            "force_llm": parsed_args.force_llm,
            "llm_model": parsed_args.model,
        }
        return workflow_runner.run_single(raw_cli_data, app_config)


def _app_config(data: dict) -> dict:
    """
    Собирает настройки запуска графа из входного словаря
    Используется всеми сервисами, которые обращаются к WorkflowRunner.
    """
    return {
        "require_llm": bool(data.get("require_llm", False)),
        "force_llm": bool(data.get("force_llm", False)),
        "llm_model": data.get("llm_model", "qwen2.5:7b-instruct"),
    }


def _serialize_case(case) -> Dict[str, Any]:
    """
    Преобразует ORM-модель кейса в JSON-словарь для фронтенда
    Даты переводятся в ISO-строки, чтобы браузер мог их корректно читать.
    """
    return {
        "id": case.id,
        "patient_id": case.patient_id,
        "title": case.title,
        "status": case.status,
        "current_stage": case.current_stage,
        "latest_risk": case.latest_risk,
        "latest_risk_level": case.latest_risk_level,
        "latest_triage_category": case.latest_triage_category,
        "latest_explanation": case.latest_explanation,
        "latest_payload": case.latest_payload,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None,
        "closed_at": case.closed_at.isoformat() if case.closed_at else None,
    }


class CatalogService:
    """
    Сервис для выдачи медицинского каталога на фронтенд.
    Возвращает справочники анализов, исследований, процедур, препаратов и диагнозов.
    """

    @staticmethod
    def payload() -> dict:
        """Возвращает весь каталог в формате JSON-словаря."""
        return catalog_as_json()


class CaseService:
    """
    Сервис для управления стационарными кейсами пациента.
    Кейс связан с пациентом напрямую, а анализы и исследования связаны уже с кейсом.
    """

    def __init__(self, db_session: Session):
        # :param db_session: Активная сессия SQLAlchemy для работы с кейсами.
        self.repo = sql_database_repository(db_session)

    def start_case(self, data: dict) -> dict:
        """
        Создает новый стационарный кейс пациента
        """
        patient_id = data.get("patient_id")
        raw_data = {
            k: v for k, v in data.items()
            if k not in {"patient_id", "require_llm", "force_llm", "llm_model", "reuse_active"}
        }
        case = self.repo.create_case(
            patient_id=patient_id,
            title=build_case_title(raw_data),
            llm_model=str(data.get("llm_model", "")),
            initial_payload=raw_data,
            latest_payload=raw_data,
        )
        return {
            "case_id": case.id,
            "case_status": case.status,
            "current_stage": case.current_stage,
            "case": _serialize_case(case),
        }

    def resume_case(self, case_id: str, data: dict) -> dict:
        """
        Продолжает кейс новыми наблюдениями
        Нормализует входные наблюдения и запускает повторную оценку состояния.
        """
        observations = normalize_observations(data.get("observations", []))
        return workflow_runner.resume_case(case_id, _app_config(data), self.repo, observations=observations)

    def get_case(self, case_id: str) -> dict:
        """
        Возвращает полную карточку кейса
        В ответ входят наблюдения, анализы, исследования, назначения, диагнозы и отчеты.
        """
        case = self.repo.get_case(case_id)
        if not case:
            return {"error": "Кейс не найден"}
        observations = self.repo.get_case_observations(case_id)
        assessments = self.repo.get_case_assessments(case_id)
        reports = self.repo.get_case_reports(case_id)
        studies = self.repo.get_case_studies(case_id)
        procedures = self.repo.get_case_procedures(case_id)
        medications = self.repo.get_case_medications(case_id)
        diagnoses = self.repo.get_case_diagnoses(case_id)
        protocol, tracking, summary = build_case_control(
            observations=observations,
            studies=studies,
            procedures=procedures,
            medications=medications,
            diagnoses=diagnoses,
            case_payload=case.latest_payload or case.initial_payload or {},
            latest_result={
                "triage_category": case.latest_triage_category,
                "risk_level": case.latest_risk_level,
            },
            case_started_at=case.created_at,
        )
        return {
            "case": _serialize_case(case),
            "observations": [_serialize_observation(obs) for obs in observations],
            "studies": [_serialize_study(item) for item in studies],
            "procedures": [_serialize_procedure(item) for item in procedures],
            "medications": [_serialize_medication(item) for item in medications],
            "diagnoses": [_serialize_diagnosis(item) for item in diagnoses],
            "assessments": [
                {
                    "id": item.id,
                    "run_kind": item.run_kind,
                    "risk": item.risk,
                    "risk_level": item.risk_level,
                    "triage_category": item.triage_category,
                    "next_step": item.next_step,
                    "route_reason": item.route_reason,
                    "explanation": item.explanation,
                    "created_at": item.created_at.isoformat(),
                }
                for item in assessments
            ],
            "reports": [
                {
                    "id": report.id,
                    "report_type": report.report_type,
                    "content": report.content,
                    "created_at": report.created_at.isoformat(),
                }
                for report in reports
            ],
            "time_series": build_series(observations),
            "protocol": protocol_summary(protocol),
            "tracking": tracking,
            "control_summary": summary,
        }

    def get_active(self, patient_id: int | None) -> dict:
        """
        Ищет активный кейс пациента
        Нужен для быстрого восстановления незакрытой госпитализации.
        """
        case = self.repo.get_active_case(patient_id)
        if case is None:
            return {"case": None}
        return {"case": _serialize_case(case)}

    def close_case(self, case_id: str) -> dict:
        """
        Закрывает кейс пациента
        Переводит кейс в завершенный статус без удаления данных.
        """
        case = self.repo.close_case(case_id)
        if case is None:
            return {"error": "Кейс не найден"}
        return {"case": _serialize_case(case)}

    def reopen_case(self, case_id: str) -> dict:
        """
        Переоткрывает ранее закрытый кейс
        Возвращает его в активный статус для дальнейшего наблюдения.
        """
        case = self.repo.reopen_case(case_id)
        if case is None:
            return {"error": "Кейс не найден"}
        return {"case": _serialize_case(case)}

    def delete_case(self, case_id: str) -> dict:
        """
        Удаляет кейс и все связанные с ним медицинские данные
        Используется отдельной кнопкой удаления на карточке кейса.
        """
        if not self.repo.delete_case(case_id):
            return {"error": "Кейс не найден"}
        return {"success": True}

    def generate_report(self, case_id: str, data: dict) -> dict:
        """
        Генерирует клинический отчет или эпикриз по кейсу
        Отчет строится на последних данных кейса и временных рядах наблюдений.
        """
        return workflow_runner.generate_case_report(case_id, _app_config(data), self.repo)

    def get_control_dashboard(self, case_id: str) -> dict:
        """
        Собирает контрольную панель кейса
        Рассчитывает протокол, трекинг выполнения, готовность и предупреждения.
        """
        case = self.repo.get_case(case_id)
        if not case:
            return {"error": "Кейс не найден"}
        observations = self.repo.get_case_observations(case_id)
        studies = self.repo.get_case_studies(case_id)
        procedures = self.repo.get_case_procedures(case_id)
        medications = self.repo.get_case_medications(case_id)
        diagnoses = self.repo.get_case_diagnoses(case_id)
        protocol, tracking, summary = build_case_control(
            observations=observations,
            studies=studies,
            procedures=procedures,
            medications=medications,
            diagnoses=diagnoses,
            case_payload=case.latest_payload or case.initial_payload or {},
            latest_result={
                "triage_category": case.latest_triage_category,
                "risk_level": case.latest_risk_level,
            },
            case_started_at=case.created_at,
        )
        return {
            "case": _serialize_case(case),
            "protocol": protocol_summary(protocol),
            "tracking": tracking,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Сериализаторы данных для фронтенда
# ---------------------------------------------------------------------------
def _iso(value):
    """Преобразует дату в ISO-строку или возвращает None."""
    return value.isoformat() if value else None


def _serialize_observation(obs) -> Dict[str, Any]:
    """
    Преобразует наблюдение или анализ в JSON-словарь
    Дополнительно подставляет референсные значения из медицинского каталога.
    """
    ref_low = None
    ref_high = None
    if obs.code and obs.category == "lab" and obs.code in LAB_BY_CODE:
        ref_low = LAB_BY_CODE[obs.code].ref_low
        ref_high = LAB_BY_CODE[obs.code].ref_high
    elif obs.code and obs.category == "vital" and obs.code in VITAL_BY_CODE:
        ref_low = VITAL_BY_CODE[obs.code].ref_low
        ref_high = VITAL_BY_CODE[obs.code].ref_high
    return {
        "id": obs.id,
        "case_id": obs.case_id,
        "category": obs.category,
        "code": obs.code,
        "name": obs.name,
        "value_num": obs.value_num,
        "value_text": obs.value_text,
        "unit": obs.unit,
        "flag": obs.flag,
        "source": obs.source,
        "note": obs.note,
        "recorded_at": _iso(obs.recorded_at),
        "ref_low": ref_low,
        "ref_high": ref_high,
    }


def _serialize_study(item) -> Dict[str, Any]:
    """Преобразует инструментальное исследование в JSON-словарь."""
    return {
        "id": item.id,
        "case_id": item.case_id,
        "code": item.code,
        "name": item.name,
        "status": item.status,
        "started_at": _iso(item.started_at),
        "completed_at": _iso(item.completed_at),
        "result_text": item.result_text,
        "result_json": item.result_json,
        "priority": item.priority,
        "ordered_by": item.ordered_by,
        "note": item.note,
        "created_at": _iso(item.created_at),
        "updated_at": _iso(item.updated_at),
    }


def _serialize_procedure(item) -> Dict[str, Any]:
    """Преобразует процедуру или вмешательство в JSON-словарь."""
    return {
        "id": item.id,
        "case_id": item.case_id,
        "code": item.code,
        "name": item.name,
        "status": item.status,
        "started_at": _iso(item.started_at),
        "completed_at": _iso(item.completed_at),
        "operator": item.operator,
        "details_json": item.details_json,
        "priority": item.priority,
        "note": item.note,
        "created_at": _iso(item.created_at),
        "updated_at": _iso(item.updated_at),
    }


def _serialize_medication(item) -> Dict[str, Any]:
    """Преобразует назначение препарата в JSON-словарь."""
    return {
        "id": item.id,
        "case_id": item.case_id,
        "code": item.code,
        "name": item.name,
        "med_class": item.med_class,
        "dose": item.dose,
        "unit": item.unit,
        "route": item.route,
        "frequency": item.frequency,
        "started_at": _iso(item.started_at),
        "stopped_at": _iso(item.stopped_at),
        "status": item.status,
        "prescribed_by": item.prescribed_by,
        "note": item.note,
        "created_at": _iso(item.created_at),
        "updated_at": _iso(item.updated_at),
    }


def _serialize_diagnosis(item) -> Dict[str, Any]:
    """Преобразует диагноз в JSON-словарь."""
    return {
        "id": item.id,
        "case_id": item.case_id,
        "icd10": item.icd10,
        "name": item.name,
        "diagnosis_type": item.diagnosis_type,
        "established_at": _iso(item.established_at),
        "note": item.note,
        "created_at": _iso(item.created_at),
    }


# ---------------------------------------------------------------------------
# CRUD-сервисы, которые используют медицинский каталог
# ---------------------------------------------------------------------------
class ObservationService:
    """
    Сервис для работы с витальными показателями и лабораторными анализами.
    Все записи привязаны к конкретному стационарному кейсу и имеют время записи.
    """

    def __init__(self, db_session: Session):
        # :param db_session: Активная сессия SQLAlchemy для работы с наблюдениями.
        self.repo = sql_database_repository(db_session)

    def list_vitals(self, case_id: str) -> list:
        """Возвращает все витальные показатели выбранного кейса."""
        return [
            _serialize_observation(obs)
            for obs in self.repo.get_case_observations(case_id)
            if obs.category == "vital"
        ]

    def list_labs(self, case_id: str) -> list:
        """Возвращает все лабораторные анализы выбранного кейса."""
        return [
            _serialize_observation(obs)
            for obs in self.repo.get_case_observations(case_id)
            if obs.category == "lab"
        ]

    def add_vital(self, case_id: str, data: dict) -> dict:
        """
        Добавляет новый витальный показатель
        Код показателя сверяется с каталогом, а флаг нормы рассчитывается автоматически.
        """
        code = str(data.get("code", "")).strip()
        if code not in VITAL_BY_CODE:
            return {"error": f"Неизвестный код витального показателя: {code}"}
        value_num = _to_float(data.get("value_num", data.get("value")))
        definition = VITAL_BY_CODE[code]
        flag = flag_for_vital(code, value_num) if value_num is not None else "unknown"
        obs = self.repo.add_case_observations(case_id, [
            {
                "category": "vital",
                "code": code,
                "name": definition.name_ru,
                "value_num": value_num,
                "value_text": data.get("value_text"),
                "unit": definition.unit,
                "flag": flag,
                "source": data.get("source", "manual"),
                "note": data.get("note", ""),
                "recorded_at": data.get("recorded_at"),
            }
        ])
        return _serialize_observation(obs[0])

    def add_lab(self, case_id: str, data: dict) -> dict:
        """
        Добавляет новый лабораторный анализ
        Код анализа сверяется с каталогом, а флаг нормы рассчитывается автоматически.
        """
        code = str(data.get("code", "")).strip()
        if code not in LAB_BY_CODE:
            return {"error": f"Неизвестный код анализа: {code}"}
        value_num = _to_float(data.get("value_num", data.get("value")))
        value_text = data.get("value_text")
        definition = LAB_BY_CODE[code]
        flag = flag_for_lab(code, value_num) if value_num is not None else "unknown"
        obs = self.repo.add_case_observations(case_id, [
            {
                "category": "lab",
                "code": code,
                "name": definition.name_ru,
                "value_num": value_num,
                "value_text": value_text,
                "unit": data.get("unit") or definition.unit,
                "flag": flag,
                "source": data.get("source", "manual"),
                "note": data.get("note", ""),
                "recorded_at": data.get("recorded_at"),
            }
        ])
        return _serialize_observation(obs[0])

    def update(self, observation_id: int, data: dict) -> dict:
        """
        Обновляет витальный показатель или анализ
        При изменении числового значения пересчитывает флаг нормы.
        """
        fields = dict(data)
        if "value_num" in fields:
            fields["value_num"] = _to_float(fields.get("value_num"))
        obs = self.repo.update_case_observation(observation_id, **fields)
        if obs is None:
            return {"error": "Наблюдение не найдено"}
        # Пересчитываем флаг, если изменилось значение показателя
        if obs.value_num is not None and obs.code:
            if obs.category == "lab":
                new_flag = flag_for_lab(obs.code, float(obs.value_num))
            elif obs.category == "vital":
                new_flag = flag_for_vital(obs.code, float(obs.value_num))
            else:
                new_flag = obs.flag
            if new_flag != obs.flag:
                obs = self.repo.update_case_observation(observation_id, flag=new_flag)
        return _serialize_observation(obs)

    def delete(self, observation_id: int) -> dict:
        """Удаляет витальный показатель или анализ по ID."""
        if not self.repo.delete_case_observation(observation_id):
            return {"error": "Наблюдение не найдено"}
        return {"success": True}


class StudyService:
    """
    Сервис для работы с инструментальными исследованиями.
    Создает, обновляет, удаляет и возвращает исследования, привязанные к кейсу.
    """

    def __init__(self, db_session: Session):
        # :param db_session: Активная сессия SQLAlchemy для работы с исследованиями.
        self.repo = sql_database_repository(db_session)

    def list(self, case_id: str) -> list:
        """Возвращает список исследований выбранного кейса."""
        return [_serialize_study(item) for item in self.repo.get_case_studies(case_id)]

    def add(self, case_id: str, data: dict) -> dict:
        """
        Добавляет инструментальное исследование в кейс
        Название берется из каталога, если пользователь передал только код.
        """
        code = str(data.get("code", "")).strip()
        definition = STUDY_BY_CODE.get(code)
        name = data.get("name") or (definition.name_ru if definition else code)
        item = self.repo.add_case_study(
            case_id,
            code=code,
            name=name,
            status=data.get("status", "ordered"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result_text=data.get("result_text", ""),
            result_json=data.get("result_json", {}),
            ordered_by=data.get("ordered_by", ""),
            priority=data.get("priority", "medium"),
            note=data.get("note", ""),
        )
        return _serialize_study(item)

    def update(self, item_id: int, data: dict) -> dict:
        """Обновляет данные инструментального исследования."""
        item = self.repo.update_case_study(item_id, **data)
        if item is None:
            return {"error": "Исследование не найдено"}
        return _serialize_study(item)

    def delete(self, item_id: int) -> dict:
        """Удаляет инструментальное исследование по ID."""
        if not self.repo.delete_case_study(item_id):
            return {"error": "Исследование не найдено"}
        return {"success": True}


class ProcedureService:
    """
    Сервис для работы с процедурами и вмешательствами.
    Все процедуры относятся к конкретному стационарному кейсу.
    """

    def __init__(self, db_session: Session):
        # :param db_session: Активная сессия SQLAlchemy для работы с процедурами.
        self.repo = sql_database_repository(db_session)

    def list(self, case_id: str) -> list:
        """Возвращает список процедур выбранного кейса."""
        return [_serialize_procedure(item) for item in self.repo.get_case_procedures(case_id)]

    def add(self, case_id: str, data: dict) -> dict:
        """
        Добавляет процедуру или вмешательство
        Название подставляется из каталога, если для кода есть справочная запись.
        """
        code = str(data.get("code", "")).strip()
        definition = PROCEDURE_BY_CODE.get(code)
        name = data.get("name") or (definition.name_ru if definition else code)
        item = self.repo.add_case_procedure(
            case_id,
            code=code,
            name=name,
            status=data.get("status", "ordered"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            operator=data.get("operator", ""),
            details_json=data.get("details_json", {}),
            priority=data.get("priority", "medium"),
            note=data.get("note", ""),
        )
        return _serialize_procedure(item)

    def update(self, item_id: int, data: dict) -> dict:
        """Обновляет процедуру или вмешательство."""
        item = self.repo.update_case_procedure(item_id, **data)
        if item is None:
            return {"error": "Процедура не найдена"}
        return _serialize_procedure(item)

    def delete(self, item_id: int) -> dict:
        """Удаляет процедуру или вмешательство по ID."""
        if not self.repo.delete_case_procedure(item_id):
            return {"error": "Процедура не найдена"}
        return {"success": True}


class MedicationService:
    """
    Сервис для работы с медикаментозными назначениями.
    При добавлении может подставлять класс, дозу, единицу и путь введения из каталога.
    """

    def __init__(self, db_session: Session):
        # :param db_session: Активная сессия SQLAlchemy для работы с назначениями.
        self.repo = sql_database_repository(db_session)

    def list(self, case_id: str) -> list:
        """Возвращает список назначений выбранного кейса."""
        return [_serialize_medication(item) for item in self.repo.get_case_medications(case_id)]

    def add(self, case_id: str, data: dict) -> dict:
        """
        Добавляет медикаментозное назначение
        Если часть полей не передана, заполняет их типовыми значениями из каталога.
        """
        code = str(data.get("code", "")).strip()
        definition = MEDICATION_BY_CODE.get(code)
        name = data.get("name") or (definition.name_ru if definition else code)
        med_class = data.get("med_class") or (definition.group if definition else "")
        dose = data.get("dose") or (definition.typical_dose if definition else "")
        unit = data.get("unit") or (definition.typical_unit if definition else "")
        route = data.get("route") or (definition.default_route if definition else "po")
        item = self.repo.add_case_medication(
            case_id,
            code=code,
            name=name,
            med_class=med_class,
            dose=dose,
            unit=unit,
            route=route,
            frequency=data.get("frequency", ""),
            started_at=data.get("started_at"),
            stopped_at=data.get("stopped_at"),
            status=data.get("status", "active"),
            prescribed_by=data.get("prescribed_by", ""),
            note=data.get("note", ""),
        )
        return _serialize_medication(item)

    def update(self, item_id: int, data: dict) -> dict:
        """Обновляет медикаментозное назначение."""
        item = self.repo.update_case_medication(item_id, **data)
        if item is None:
            return {"error": "Назначение не найдено"}
        return _serialize_medication(item)

    def delete(self, item_id: int) -> dict:
        """Удаляет медикаментозное назначение по ID."""
        if not self.repo.delete_case_medication(item_id):
            return {"error": "Назначение не найдено"}
        return {"success": True}


class DiagnosisService:
    """
    Сервис для работы с диагнозами кейса.
    Использует каталог МКБ-10 для подстановки названия диагноза по коду.
    """

    def __init__(self, db_session: Session):
        # :param db_session: Активная сессия SQLAlchemy для работы с диагнозами.
        self.repo = sql_database_repository(db_session)

    def list(self, case_id: str) -> list:
        """Возвращает список диагнозов выбранного кейса."""
        return [_serialize_diagnosis(item) for item in self.repo.get_case_diagnoses(case_id)]

    def add(self, case_id: str, data: dict) -> dict:
        """
        Добавляет диагноз в кейс
        Код МКБ-10 нормализуется к верхнему регистру, название берется из каталога.
        """
        icd10 = str(data.get("icd10", "")).strip().upper()
        definition = DIAGNOSIS_BY_ICD.get(icd10)
        name = data.get("name") or (definition.name_ru if definition else icd10)
        item = self.repo.add_case_diagnosis(
            case_id,
            icd10=icd10,
            name=name,
            diagnosis_type=data.get("diagnosis_type", "primary"),
            established_at=data.get("established_at"),
            note=data.get("note", ""),
        )
        return _serialize_diagnosis(item)

    def update(self, item_id: int, data: dict) -> dict:
        """Обновляет диагноз."""
        item = self.repo.update_case_diagnosis(item_id, **data)
        if item is None:
            return {"error": "Диагноз не найден"}
        return _serialize_diagnosis(item)

    def delete(self, item_id: int) -> dict:
        """Удаляет диагноз по ID."""
        if not self.repo.delete_case_diagnosis(item_id):
            return {"error": "Диагноз не найден"}
        return {"success": True}


class ReassessService:
    """
    Сервис для повторной оценки стационарного кейса.
    Запускает WorkflowRunner на уже накопленных данных кейса без добавления новых наблюдений.
    """

    def __init__(self, db_session: Session):
        # :param db_session: Активная сессия SQLAlchemy для повторной оценки.
        self.repo = sql_database_repository(db_session)

    def run(self, case_id: str, data: dict | None = None) -> dict:
        """Запускает переоценку риска и обновляет состояние кейса."""
        return workflow_runner.resume_case(
            case_id,
            _app_config(data or {}),
            self.repo,
            observations=[],
        )


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _to_float(value) -> float | None:
    """
    Безопасно преобразует входное значение в float
    Поддерживает строки с запятой и возвращает None для пустых или неверных значений.
    """
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str):
            return float(value.replace(",", "."))
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "PatientService", "TriageService", "CaseService", "CatalogService",
    "ObservationService", "StudyService", "ProcedureService",
    "MedicationService", "DiagnosisService", "ReassessService",
]
