from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.WorkflowRunner import workflow_runner
from src.evaluation.metrics import auc_roc, sensitivity, specificity
from src.medical.rules import evaluate_hard_rules


def _risk_to_binary(risk: float, threshold: float = 0.5) -> int:
    return 1 if risk >= threshold else 0


def evaluate_dataset(dataset: List[Dict[str, Any]], *, model_name: str, threshold: float = 0.5) -> Dict[str, Any]:
    y_true: List[int] = []
    y_score_model: List[float] = []
    y_pred_model: List[int] = []
    y_score_baseline: List[float] = []
    y_pred_baseline: List[int] = []

    for item in dataset:
        patient_data = dict(item.get("patient_data", {}))
        label = int(item.get("label", 0))
        result = workflow_runner.run_single(
            patient_data,
            {"require_llm": False, "force_llm": False, "llm_model": model_name},
        )
        baseline_risk, _, _, _ = evaluate_hard_rules(patient_data)

        y_true.append(label)
        y_score_model.append(float(result.get("risk", 0.0)))
        y_pred_model.append(_risk_to_binary(float(result.get("risk", 0.0)), threshold))
        y_score_baseline.append(float(baseline_risk))
        y_pred_baseline.append(_risk_to_binary(float(baseline_risk), threshold))

    return {
        "cases": len(dataset),
        "threshold": threshold,
        "model": {
            "sensitivity": sensitivity(y_true, y_pred_model),
            "specificity": specificity(y_true, y_pred_model),
            "auc": auc_roc(y_true, y_score_model),
        },
        "baseline_rules": {
            "sensitivity": sensitivity(y_true, y_pred_baseline),
            "specificity": specificity(y_true, y_pred_baseline),
            "auc": auc_roc(y_true, y_score_baseline),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ACS workflow against labeled dataset")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset with patient_data and label")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="data/eval_result.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    result = evaluate_dataset(dataset, model_name=args.model, threshold=args.threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
