# src/core/WorkflowRunner.py
from typing import Dict, Any

from src.core.case_runtime import (
    build_case_summary,
    build_case_title,
    build_series,
    infer_case_status,
    make_report_query,
    merge_payload_with_case,
    merge_payload_with_observations,
    normalize_observations,
)
from src.core.graph import graph
from src.core.patient_control import build_case_control, has_pending_critical
from src.core.report_graph import report_graph

"""
Памятка по добавлению нового свойства для парсинга:
1. CLIParser._setup_patient_args()
2. PATIENT_FIELDS в классе data_payload_Builder
3. REQUIRED_FIELDS по необходимости
"""

class data_payload_builder:
    """Трансформирует сырые словари в payload для графа"""
    
    PATIENT_FIELDS = [
        "name", "pain_type", "ecg_changes", "troponin", "hr", "bp", 
        "age", "gender", "spo2", "glucose", "creatinine", "killip_class", 
        "echo_dkg_results", "admission_time", "pain_onset_time", "symptoms_text"
    ]
    REQUIRED_FIELDS = ["name", "pain_type", "ecg_changes", "troponin", "hr", "bp"]

    @classmethod
    def build(cls, raw_data: Dict[str, Any], app_config: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает итоговый словарь для отправки в граф"""

        patient_data = {
            field: raw_data.get(field) 
            for field in cls.PATIENT_FIELDS 
            if raw_data.get(field) is not None
        }

        has_structured = all(req_field in patient_data for req_field in cls.REQUIRED_FIELDS)
        free_text = str(raw_data.get("free_text", "")).strip()

        if not has_structured and not free_text:
            raise ValueError(
                f"Ошибка: Передайте либо полный набор полей ({', '.join(cls.REQUIRED_FIELDS)}), "
                f"либо используйте параметр --free-text."
            )

        return {
            "patient_data": patient_data if has_structured else {},
            "free_text": free_text,
            "require_llm": app_config.get("require_llm", False),
            "force_llm": app_config.get("force_llm", False),
            "llm_model": app_config.get("llm_model"),
        }


class workflow_runner:
    """Обертка над вызовом LangGraph"""

    @staticmethod
    def run_single(raw_data: Dict[str, Any], app_config: Dict[str, Any]) -> dict:
        payload = data_payload_builder.build(raw_data, app_config)
        result = graph.invoke(payload)
        return workflow_runner._format_result(result, app_config)

    @staticmethod
    def start_case(
        raw_data: Dict[str, Any],
        app_config: Dict[str, Any],
        repository: Any,
        *,
        patient_id: int | None = None,
        reuse_active: bool = True,
    ) -> dict:
        if reuse_active and patient_id is not None:
            existing = repository.get_active_case(patient_id)
            if existing is not None:
                return {
                    "case_id": existing.id,
                    "case_status": existing.status,
                    "current_stage": existing.current_stage,
                    "reused_existing": True,
                }
        payload = data_payload_builder.build(raw_data, app_config)
        latest_payload = dict(payload.get("patient_data", {}))
        case = repository.create_case(
            patient_id=patient_id,
            title=build_case_title(latest_payload),
            llm_model=str(app_config.get("llm_model", "")),
            initial_payload=latest_payload,
            latest_payload=latest_payload,
        )
        result = workflow_runner._format_result(graph.invoke(payload), app_config)
        protocol_pending = workflow_runner._protocol_pending(repository, case.id, latest_payload, result)
        status, current_stage = infer_case_status(result, protocol_pending=protocol_pending)
        repository.save_case_assessment(
            case_id=case.id,
            run_kind="initial",
            payload_snapshot=latest_payload,
            result=result,
        )
        repository.update_case_state(
            case_id=case.id,
            latest_payload=latest_payload,
            result=result,
            status=status,
            current_stage=current_stage,
        )
        return {
            "case_id": case.id,
            "case_status": status,
            "current_stage": current_stage,
            **result,
        }

    @staticmethod
    def resume_case(
        case_id: str,
        app_config: Dict[str, Any],
        repository: Any,
        *,
        observations: list[Dict[str, Any]] | None = None,
    ) -> dict:
        case = repository.get_case(case_id)
        if case is None:
            raise ValueError("Case not found")

        normalized_observations = normalize_observations(observations or [])
        if normalized_observations:
            repository.add_case_observations(case_id, normalized_observations)

        all_observations = repository.get_case_observations(case_id)
        observation_payload = [
            {
                "category": obs.category,
                "code": getattr(obs, "code", ""),
                "name": obs.name,
                "value_num": obs.value_num,
                "value_text": obs.value_text,
                "unit": obs.unit,
                "flag": getattr(obs, "flag", "unknown"),
                "source": obs.source,
                "recorded_at": obs.recorded_at.isoformat(),
            }
            for obs in all_observations
        ]
        studies = repository.get_case_studies(case_id)
        procedures = repository.get_case_procedures(case_id)
        medications = repository.get_case_medications(case_id)
        diagnoses = repository.get_case_diagnoses(case_id)
        base_payload = dict(case.latest_payload or case.initial_payload or {})
        merged_payload = merge_payload_with_case(
            base_payload,
            observations=observation_payload,
            medications=medications,
            diagnoses=diagnoses,
            studies=studies,
            procedures=procedures,
        )
        raw_data = dict(merged_payload)
        raw_data["free_text"] = ""
        result = workflow_runner.run_single(raw_data, app_config)
        protocol_pending = workflow_runner._protocol_pending(repository, case_id, merged_payload, result)
        status, current_stage = infer_case_status(result, protocol_pending=protocol_pending)
        repository.save_case_assessment(
            case_id=case_id,
            run_kind="dynamic_reassessment" if normalized_observations else "resume_case",
            payload_snapshot=merged_payload,
            result=result,
        )
        repository.update_case_state(
            case_id=case_id,
            latest_payload=merged_payload,
            result=result,
            status=status,
            current_stage=current_stage,
        )
        return {
            "case_id": case_id,
            "case_status": status,
            "current_stage": current_stage,
            "time_series": build_series(all_observations),
            **result,
        }

    @staticmethod
    def generate_case_report(case_id: str, app_config: Dict[str, Any], repository: Any) -> dict:
        case = repository.get_case(case_id)
        if case is None:
            raise ValueError("Case not found")

        observations = repository.get_case_observations(case_id)
        assessments = repository.get_case_assessments(case_id)
        latest_result = {
            "risk": case.latest_risk or 0.0,
            "risk_level": case.latest_risk_level,
            "triage_category": case.latest_triage_category,
            "next_step": case.latest_next_step,
            "explanation": case.latest_explanation,
            "citations": case.latest_citations or [],
        }
        series = build_series(observations)
        case_summary = build_case_summary(case.latest_payload or case.initial_payload or {}, latest_result, series)
        report_state = report_graph.invoke(
            {
                "case_summary": case_summary,
                "report_query": make_report_query(case.latest_payload or case.initial_payload or {}, latest_result, series),
                "llm_model": app_config.get("llm_model", ""),
                "require_llm": app_config.get("require_llm", False),
            }
        )
        report = repository.save_case_report(
            case_id=case_id,
            report_type="epicrisis",
            content=str(report_state.get("report", "")),
            citations=list(report_state.get("citations", [])),
        )
        return {
            "case_id": case_id,
            "report_id": report.id,
            "report_type": report.report_type,
            "content": report.content,
            "citations": report.citations_json,
            "assessment_runs": len(assessments),
        }

    @staticmethod
    def _protocol_pending(repository: Any, case_id: str, payload: Dict[str, Any], result: Dict[str, Any]) -> bool:
        try:
            observations = repository.get_case_observations(case_id)
            studies = repository.get_case_studies(case_id)
            procedures = repository.get_case_procedures(case_id)
            medications = repository.get_case_medications(case_id)
            diagnoses = repository.get_case_diagnoses(case_id)
        except Exception:
            # PostgreSQL: после ошибки SQL сессия 'отравлена' - без rollback следующий INSERT даст InFailedSqlTransaction.
            sess = getattr(repository, "session", None)
            if sess is not None:
                try:
                    sess.rollback()
                except Exception:
                    pass
            return False
        case = repository.get_case(case_id)
        case_started = getattr(case, "created_at", None) if case else None
        _, tracking, _ = build_case_control(
            observations=observations,
            studies=studies,
            procedures=procedures,
            medications=medications,
            diagnoses=diagnoses,
            case_payload=payload,
            latest_result=result,
            case_started_at=case_started,
        )
        return has_pending_critical(tracking)

    @staticmethod
    def _format_result(result: Dict[str, Any], app_config: Dict[str, Any]) -> dict:
        return {
            "model": app_config.get("llm_model"),
            "risk": round(float(result.get("risk", 0.0)), 3),
            "risk_level": result.get("risk_level"),
            "explanation": result.get("explanation"),
            "citations": result.get("citations", []),
            "record_id": result.get("save_id"),
            "llm_used": bool(result.get("llm_used", False)),
            "parse_confidence": result.get("parse_confidence"),
            "missing_fields": result.get("missing_fields", []),
            "route_confidence": result.get("route_confidence"),
            "next_step": result.get("next_step"),
            "triage_category": result.get("triage_category"),
            "route_reason": result.get("route_reason"),
            "acs_diagnosis": result.get("acs_diagnosis"),
            "rule_fires": result.get("rule_fires", []),
            "rule_reasons": result.get("rule_reasons", []),
            "node_trace": result.get("node_trace", []),
        }
