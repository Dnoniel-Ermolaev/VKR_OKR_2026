from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

from src.core.graph import graph

"""
Памятка по добавлению нового свойства для парсинга:
1. CLIParser._setup_patient_args()
2. PATIENT_FIELDS в классе data_payload_Builder
3. REQUIRED_FIELDS по необходимости
"""

class CLIParser:

    """Парсим аргументы из командной строки"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Console runner for ACS diagnosis system")
        self._setup_app_args()
        self._setup_patient_args()

    def _setup_app_args(self):
        """Настройки самого приложения (модели, файлы, режимы)"""

        parser_group = self.parser.add_argument_group("App Configuration")
        parser_group.add_argument(
            "--mode",
            choices=["single", "ab"],
            default="single",
            help="single: one model, ab: compare two models on one case",
        )
        parser_group.add_argument(
            "--model", 
            default=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            help="Primary Ollama model for LLM assessment"
        )
        parser_group.add_argument(
            "--model-b", 
            default=os.getenv("OLLAMA_MODEL_B", "qwen2.5:3b-instruct"),
            help="Second model for A/B testing"
        )
        parser_group.add_argument(
            "--output",
            default="result.json",
            help="Path to output JSON file (default: result.json)",
        )
        parser_group.add_argument(
            "--require-llm",
            action="store_true",
            help="Fail if LLM/Ollama is unavailable instead of using fallback",
        )
        parser_group.add_argument(
            "--force-llm",
            action="store_true",
            help="Always run LLM branch even for obvious high-risk rule cases",
        )

    def _setup_patient_args(self):
        """Медицинские данные пациента"""

        parser_group = self.parser.add_argument_group("Patient Data")
        # Имена аргументов должны совпадать с ключами Pydantic-модели PatientData!
        parser_group.add_argument("--name")
        parser_group.add_argument("--age", type=int)
        parser_group.add_argument("--pain-type", choices=["typical", "atypical", "none"])
        parser_group.add_argument("--troponin", type=float)
        parser_group.add_argument("--ecg-changes")
        parser_group.add_argument("--hr", type=int)
        parser_group.add_argument("--bp", help="Format systolic/diastolic, e.g. 120/80")
        parser_group.add_argument("--gender", choices=["male", "female", "unknown"], default="unknown")
        parser_group.add_argument("--spo2", type=float)
        parser_group.add_argument("--glucose", type=float)
        parser_group.add_argument("--creatinine", type=float)
        parser_group.add_argument("--killip-class", default="")
        parser_group.add_argument("--echo-dkg-results", default="")
        parser_group.add_argument("--admission-time", default="")
        parser_group.add_argument("--pain-onset-time", default="")
        parser_group.add_argument("--symptoms-text", default="")
        parser_group.add_argument("--free-text", default="")

    def parse(self) -> argparse.Namespace:
        return self.parser.parse_args()


class data_payload_Builder:
    """Трансформирует сырые аргументы в payload для LangGraph"""
    
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
    """Исполнитель"""

    @staticmethod
    def run_single(raw_data: Dict[str, Any], app_config: Dict[str, Any]) -> dict:
        payload = data_payload_Builder.build(raw_data, app_config)
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


def main() -> None:

    cli_parser = CLIParser()
    args = cli_parser.parse()

    raw_data = vars(args)
    app_config = {
        "require_llm": args.require_llm,
        "force_llm": args.force_llm,
    }

    try:
        if args.mode == "single":
            app_config["llm_model"] = args.model
            printable = workflow_runner.run_single(raw_data, app_config)
        else:
            app_config["llm_model"] = args.model
            result_a = workflow_runner.run_single(raw_data, app_config)
            
            app_config["llm_model"] = args.model_b
            result_b = workflow_runner.run_single(raw_data, app_config)

            printable = {
                "mode": "ab",
                "patient_name": args.name,
                "results": [result_a, result_b],
            }
    except Exception as exc:
        print(f"Ошибка выполнения: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(printable, ensure_ascii=False, indent=2)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"JSON saved to: {output_path.resolve()}")
    try:
        print(rendered)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((rendered + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
