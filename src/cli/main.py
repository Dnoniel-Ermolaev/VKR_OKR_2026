from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.core.graph import graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Console runner for ACS diagnosis prototype")
    parser.add_argument("--name", required=True)
    parser.add_argument("--pain-type", choices=["typical", "atypical", "none"], required=True)
    parser.add_argument("--ecg-changes", required=True)
    parser.add_argument("--troponin", type=float, required=True)
    parser.add_argument("--hr", type=int, required=True)
    parser.add_argument("--bp", required=True, help="Format systolic/diastolic, e.g. 120/80")
    parser.add_argument("--symptoms-text", default="")
    parser.add_argument(
        "--output",
        default="result.json",
        help="Path to output JSON file (default: result.json)",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if LLM/Ollama is unavailable instead of using fallback",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Always run LLM branch even for obvious high-risk rule cases",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "ab"],
        default="single",
        help="single: one model, ab: compare two models on one case",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        help="Primary Ollama model for LLM assessment",
    )
    parser.add_argument(
        "--model-b",
        default=os.getenv("OLLAMA_MODEL_B", "medgemma:latest"),
        help="Second model for A/B mode",
    )
    return parser


def _run_once(args: argparse.Namespace, model_name: str) -> dict:
    payload = {
        "patient_data": {
            "name": args.name,
            "pain_type": args.pain_type,
            "ecg_changes": args.ecg_changes,
            "troponin": args.troponin,
            "hr": args.hr,
            "bp": args.bp,
            "symptoms_text": args.symptoms_text,
        },
        "require_llm": args.require_llm,
        "force_llm": args.force_llm,
        "llm_model": model_name,
    }
    result = graph.invoke(payload)
    return {
        "model": model_name,
        "risk": round(float(result.get("risk", 0.0)), 3),
        "risk_level": result.get("risk_level"),
        "explanation": result.get("explanation"),
        "record_id": result.get("save_id"),
        "llm_used": bool(result.get("llm_used", False)),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.mode == "single":
            printable = _run_once(args, args.model)
        else:
            result_a = _run_once(args, args.model)
            result_b = _run_once(args, args.model_b)
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
    output_path.write_text(json.dumps(printable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON saved to: {output_path.resolve()}")
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
