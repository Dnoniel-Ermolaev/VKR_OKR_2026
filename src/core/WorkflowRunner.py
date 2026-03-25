# src/core/WorkflowRunner.py
from typing import Dict, Any
from src.core.graph import graph

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
        
        return {
            "model": app_config.get("llm_model"),
            "risk": round(float(result.get("risk", 0.0)), 3),
            "risk_level": result.get("risk_level"),
            "explanation": result.get("explanation"),
            "record_id": result.get("save_id"),
            "llm_used": bool(result.get("llm_used", False)),
            "parse_confidence": result.get("parse_confidence"),
            "missing_fields": result.get("missing_fields", []),
            "route_confidence": result.get("route_confidence"),
            "next_step": result.get("next_step"),
            "triage_category": result.get("triage_category"),
            "route_reason": result.get("route_reason"),
        }
