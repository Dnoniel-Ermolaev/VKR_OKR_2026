from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.core.WorkflowRunner import workflow_runner

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
