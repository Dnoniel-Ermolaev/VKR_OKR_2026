from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.core.graph import graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Console runner for ACS diagnosis prototype")
    parser.add_argument("--name")
    parser.add_argument("--pain-type", choices=["typical", "atypical", "none"])
    parser.add_argument("--ecg-changes")
    parser.add_argument("--troponin", type=float)
    parser.add_argument("--hr", type=int)
    parser.add_argument("--bp", help="Format systolic/diastolic, e.g. 120/80")
    parser.add_argument("--age", type=int)
    parser.add_argument("--gender", choices=["male", "female", "unknown"], default="unknown")
    parser.add_argument("--spo2", type=float)
    parser.add_argument("--glucose", type=float)
    parser.add_argument("--creatinine", type=float)
    parser.add_argument("--killip-class", default="")
    parser.add_argument("--echo-dkg-results", default="")
    parser.add_argument("--admission-time", default="")
    parser.add_argument("--pain-onset-time", default="")
    parser.add_argument("--symptoms-text", default="")
    parser.add_argument("--free-text", default="", help="Raw history text for LLM parsing")
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
        default=os.getenv("OLLAMA_MODEL_B", "qwen2.5:3b-instruct"),
        help="Second model for A/B mode",
    )
    return parser


def _has_structured_input(args: argparse.Namespace) -> bool:
    required = [args.name, args.pain_type, args.ecg_changes, args.troponin, args.hr, args.bp]
    return all(value is not None for value in required)


def _run_once(args: argparse.Namespace, model_name: str) -> dict:
    has_structured = _has_structured_input(args)
    if not has_structured and not str(args.free_text).strip():
        raise ValueError(
            "Передайте либо полный набор структурированных полей "
            "(name, pain-type, ecg-changes, troponin, hr, bp), либо --free-text."
        )

    payload = {
        "patient_data": (
            {
                "name": args.name,
                "pain_type": args.pain_type,
                "ecg_changes": args.ecg_changes,
                "troponin": args.troponin,
                "hr": args.hr,
                "bp": args.bp,
                "age": args.age,
                "gender": args.gender,
                "spo2": args.spo2,
                "glucose": args.glucose,
                "creatinine": args.creatinine,
                "killip_class": args.killip_class,
                "echo_dkg_results": args.echo_dkg_results,
                "admission_time": args.admission_time,
                "pain_onset_time": args.pain_onset_time,
                "symptoms_text": args.symptoms_text,
            }
            if has_structured
            else {}
        ),
        "free_text": str(args.free_text or ""),
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
        "parse_confidence": result.get("parse_confidence"),
        "missing_fields": result.get("missing_fields", []),
        "route_confidence": result.get("route_confidence"),
        "next_step": result.get("next_step"),
        "triage_category": result.get("triage_category"),
        "route_reason": result.get("route_reason"),
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
    rendered = json.dumps(printable, ensure_ascii=False, indent=2)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"JSON saved to: {output_path.resolve()}")
    try:
        print(rendered)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((rendered + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
