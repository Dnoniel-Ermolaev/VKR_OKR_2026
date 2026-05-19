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
from src.medical.diagnosis import AcsDiagnosis
from src.medical.protocols import PROTOCOLS, protocol_summary
from src.medical.rules import RULEBOOK
from src.medical.terminology import diagnosis_color, diagnosis_label


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
            # Явный порядок: кейсы (со всеми дочерними сущностями по cascade ORM) -> визиты -> пациент.
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
        "latest_acs_diagnosis": getattr(case, "latest_acs_diagnosis", None) or {},
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
                    "citations": item.citations_json,
                    "missing_fields": item.missing_fields_json,
                    "llm_used": item.llm_used,
                    "created_at": item.created_at.isoformat(),
                    "acs_diagnosis": getattr(item, "acs_diagnosis_json", None) or {},
                    "has_trace": bool(getattr(item, "path_trace_json", None)),
                }
                for item in assessments
            ],
            "latest_acs_diagnosis": getattr(case, "latest_acs_diagnosis", None) or {},
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


# ---------------------------------------------------------------------------
# Сервис 'Граф пациента' (Track 4 - визуализация маршрута по графу LangGraph)
# ---------------------------------------------------------------------------

# Категория узла -> CSS-класс / цвет для Cytoscape.
NODE_CATEGORY_COLORS = {
    "ingestion": "#0ea5e9",
    "router": "#a855f7",
    "rule": "#f97316",
    "diagnosis": "#ef4444",
    "triage": "#22c55e",
    "knowledge": "#14b8a6",
    "management": "#3b82f6",
    "output": "#64748b",
    "protocol": "#6366f1",
}


def _node(node_id: str, label: str, category: str, **extra: Any) -> Dict[str, Any]:
    """Сформировать описание одной вершины канонического графа."""
    return {
        "id": node_id,
        "label": label,
        "category": category,
        "color": NODE_CATEGORY_COLORS.get(category, "#94a3b8"),
        **extra,
    }


def _edge(source: str, target: str, label: str = "", kind: str = "flow") -> Dict[str, Any]:
    edge_id = f"{source}__{target}__{kind}"
    return {"id": edge_id, "source": source, "target": target, "label": label, "kind": kind}


def build_canonical_graph() -> Dict[str, Any]:
    """Собрать каноническое описание графа пациента для фронтенда.

    Объединяет:

    - узлы LangGraph (parse_input, router_pretriage, rule_check, classify_acs,
      router_diagnostic, high_risk_fast_track, low_risk_observation,
      diagnostic_uncertain, rag_retrieval, llm_assess, router_management,
      monitor_plan, recommend_treatment, output_save);
    - категории правил из RULEBOOK (свернуты в один кластер 'Правила КР');
    - диагностические метки ОКС (ИМпST, ИМбпST, ОКСпST, ОКСбпST, НС, ОКС маловероятен);
    - клинические протоколы (ИМпST / ИМбпST / НС / generic).
    """
    nodes: List[Dict[str, Any]] = [
        # ---- пайплайн анализа ----
        _node("llm_parse_history", "LLM-парсинг анамнеза", "ingestion"),
        _node("parse_input", "parse_input - нормализация payload", "ingestion"),
        _node("router_pretriage", "router_pretriage - достаточно ли данных?", "router"),
        _node("clarify_data", "clarify_data - запрос уточнений", "ingestion"),
        _node("data_quality_issue", "data_quality_issue", "output"),
        _node("rule_check", "rule_check - RULEBOOK (КР Минздрав)", "rule"),
        _node("classify_acs", "classify_acs - диагностическая метка", "diagnosis"),
        _node("router_diagnostic", "router_diagnostic", "router"),
        _node("high_risk_fast_track", "high_risk_fast_track (ИМпST/ИМбпST)", "triage"),
        _node("diagnostic_uncertain", "diagnostic_uncertain (неопределённо)", "triage"),
        _node("low_risk_observation", "low_risk_observation (НС / ОКС маловероятен)", "triage"),
        _node("rag_retrieval", "rag_retrieval - поиск по КР", "knowledge"),
        _node("llm_assess", "llm_assess - корректировка риска", "knowledge"),
        _node("router_management", "router_management", "router"),
        _node("monitor_plan", "monitor_plan - план наблюдения", "management"),
        _node("recommend_treatment", "recommend_treatment - план терапии", "management"),
        _node("output_save", "output_save - фиксация результата", "output"),
    ]
    edges: List[Dict[str, Any]] = [
        _edge("llm_parse_history", "parse_input"),
        _edge("parse_input", "router_pretriage"),
        _edge("router_pretriage", "rule_check", "достаточно данных"),
        _edge("router_pretriage", "clarify_data", "нужно уточнить"),
        _edge("clarify_data", "llm_parse_history", "retry_parse"),
        _edge("clarify_data", "data_quality_issue", "не получилось"),
        _edge("data_quality_issue", "output_save"),
        _edge("rule_check", "classify_acs"),
        _edge("classify_acs", "router_diagnostic"),
        _edge("router_diagnostic", "high_risk_fast_track", "urgent"),
        _edge("router_diagnostic", "diagnostic_uncertain", "rag_llm"),
        _edge("router_diagnostic", "low_risk_observation", "rule_only"),
        _edge("diagnostic_uncertain", "rag_retrieval"),
        _edge("rag_retrieval", "llm_assess"),
        _edge("llm_assess", "router_management"),
        _edge("high_risk_fast_track", "router_management"),
        _edge("low_risk_observation", "router_management"),
        _edge("router_management", "monitor_plan", "monitor"),
        _edge("router_management", "recommend_treatment", "recommend_treatment"),
        _edge("router_management", "output_save", "finalize"),
        _edge("monitor_plan", "output_save"),
        _edge("recommend_treatment", "output_save"),
    ]

    # ---- кластер правил RULEBOOK: одна вершина-агрегатор + по одной на категорию ----
    rule_categories = sorted({rule.category for rule in RULEBOOK})
    nodes.append(
        _node(
            "rulebook_root",
            f"RULEBOOK ({len(RULEBOOK)} правил)",
            "rule",
            description="Свод типизированных правил клинических рекомендаций Минздрава.",
        )
    )
    edges.append(_edge("rule_check", "rulebook_root", "uses", kind="reference"))
    for category in rule_categories:
        cat_id = f"rules_cat_{category}"
        nodes.append(
            _node(
                cat_id,
                {
                    "ecg": "ЭКГ-критерии",
                    "biomarker": "Биомаркеры (тропонин)",
                    "clinical": "Клиника / симптомы",
                    "hemodynamic": "Гемодинамика",
                    "score": "Скоры / демография",
                    "time": "Тайминг (окно реперфузии)",
                }.get(category, category),
                "rule",
                description=f"Категория правил: {category}",
                rule_count=sum(1 for r in RULEBOOK if r.category == category),
                rule_ids=[r.id for r in RULEBOOK if r.category == category],
            )
        )
        edges.append(_edge("rulebook_root", cat_id, kind="reference"))

    # ---- диагностические метки ----
    for diag in AcsDiagnosis:
        diag_id = f"diag_{diag.value}"
        nodes.append(
            _node(
                diag_id,
                diagnosis_label(diag.value),
                "diagnosis",
                color=diagnosis_color(diag.value),
                diagnosis_code=diag.value,
            )
        )
        edges.append(_edge("classify_acs", diag_id, kind="produces"))

    # ---- протоколы ведения ----
    for protocol in PROTOCOLS.values():
        proto_id = f"protocol_{protocol.code}"
        nodes.append(
            _node(
                proto_id,
                protocol.name,
                "protocol",
                description=protocol.description,
                protocol_code=protocol.code,
            )
        )

    return {"nodes": nodes, "edges": edges, "categories": list(NODE_CATEGORY_COLORS.keys())}


class GraphTraceService:
    """Сервис, отдающий канонический граф и трассу пациента для UI-модалки."""

    def __init__(self, db_session: Session):
        self.repo = sql_database_repository(db_session)

    def get(self, case_id: str, assessment_id: int | None = None) -> dict:
        """Вернуть граф + актуальную/историческую трассу + список ассессментов."""
        case = self.repo.get_case(case_id)
        if case is None:
            return {"error": "Кейс не найден"}

        assessments = self.repo.get_case_assessments(case_id)
        history = [
            {
                "id": item.id,
                "run_kind": item.run_kind,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "risk_level": item.risk_level,
                "triage_category": item.triage_category,
                "next_step": item.next_step,
                "acs_diagnosis": getattr(item, "acs_diagnosis_json", None) or {},
                "has_trace": bool(getattr(item, "path_trace_json", None)),
            }
            for item in assessments
        ]

        # Выбираем трассу: либо явно указанный ассессмент, либо последняя
        # сохранённая трасса в TriageCase.latest_path_trace_json.
        selected = None
        if assessment_id is not None:
            selected = next((a for a in assessments if a.id == assessment_id), None)

        latest_trace: Dict[str, Any]
        if selected is not None:
            latest_trace = dict(getattr(selected, "path_trace_json", {}) or {})
            latest_diagnosis = getattr(selected, "acs_diagnosis_json", {}) or {}
        else:
            latest_trace = dict(getattr(case, "latest_path_trace_json", {}) or {})
            latest_diagnosis = getattr(case, "latest_acs_diagnosis", {}) or {}

        if not latest_trace and assessments:
            # Фолбэк: если latest_path_trace_json пустой (например, ассессмент
            # сохранён до апгрейда), берём из последней оценки.
            last = assessments[-1]
            latest_trace = dict(getattr(last, "path_trace_json", {}) or {})
            if not latest_diagnosis:
                latest_diagnosis = getattr(last, "acs_diagnosis_json", {}) or {}

        return {
            "case": _serialize_case(case),
            "canonical_graph": build_canonical_graph(),
            "latest_trace": latest_trace,
            "latest_diagnosis": latest_diagnosis,
            "history": history,
            "selected_assessment_id": selected.id if selected is not None else None,
        }


__all__ = [
    "PatientService", "TriageService", "CaseService", "CatalogService",
    "ObservationService", "StudyService", "ProcedureService",
    "MedicationService", "DiagnosisService", "ReassessService",
    "GraphTraceService", "build_canonical_graph",
]
