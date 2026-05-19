"""Smoke-проверка интеграции классификатора ОКС/ИМ и трассы графа.

Скрипт прогоняет три синтетических сценария (соответствующих демо-кейсам
``structured_high_risk_demo_fixed.xlsx``, ``protocol_gap_demo.xlsx`` и
``patient_card_stemi_case.xlsx``) через :class:`WorkflowRunner` и печатает:

* итоговую диагностическую метку ``acs_diagnosis``;
* выбранный протокол;
* посещённые узлы LangGraph (``node_trace``);
* сработавшие правила КР (``rule_fires``).

LLM/RAG в сценариях не нужен - мы умышленно выключаем их через app_config,
чтобы тест работал автономно и быстро.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Гарантируем UTF-8 stdout, чтобы русские символы и Unicode-операторы (>=, <=)
# корректно печатались в Windows-консоли.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.WorkflowRunner import workflow_runner as WorkflowRunner


APP_CONFIG = {
    "llm_enabled": False,
    "rag_enabled": False,
}


SCENARIOS = [
    {
        "name": "patient_card_stemi_case.xlsx (ИМпST)",
        "expected_label": "im_pst",
        "patient_data": {
            "age": 64,
            "sex": "male",
            "pain_type": "typical",
            "ecg_changes": "стойкий подъём ST в V2-V5, реципрокная депрессия в II/III/aVF",
            "troponin": 8.5,
            "hr": 95,
            "bp": "100/70",
            "spo2": 94,
            "symptoms_text": "Стойкая жгучая боль за грудиной более 30 минут, иррадиация в левую руку",
        },
    },
    {
        "name": "structured_high_risk_demo_fixed.xlsx (ИМбпST/ОКСбпST)",
        "expected_label_any": ["im_bpst", "oks_bpst"],
        "patient_data": {
            "age": 71,
            "sex": "female",
            "pain_type": "typical",
            "ecg_changes": "горизонтальная депрессия ST 1.5 мм в V4-V6, инверсия T",
            "troponin": 0.06,
            "hr": 102,
            "bp": "150/95",
            "spo2": 96,
            "symptoms_text": "Боль в грудной клетке длительностью около 25 минут, потливость",
        },
    },
    {
        "name": "protocol_gap_demo.xlsx (НС / маловероятен)",
        "expected_label_any": ["ns", "oks_unlikely"],
        "patient_data": {
            "age": 58,
            "sex": "male",
            "pain_type": "atypical",
            "ecg_changes": "норма, без острых изменений",
            "troponin": 0.005,
            "hr": 78,
            "bp": "130/85",
            "spo2": 98,
            "symptoms_text": "Кратковременные эпизоды давящих болей при нагрузке",
        },
    },
]


def main() -> int:
    all_ok = True
    for scenario in SCENARIOS:
        print("\n" + "=" * 78)
        print(f"СЦЕНАРИЙ: {scenario['name']}")
        print("=" * 78)

        raw_data = dict(scenario["patient_data"])
        raw_data.setdefault("name", scenario["name"].split(" (")[0])
        raw_data["free_text"] = ""
        result = WorkflowRunner.run_single(raw_data, APP_CONFIG)
        acs = result.get("acs_diagnosis") or {}
        label = acs.get("label")
        confidence = acs.get("confidence")
        rationale = acs.get("rationale")
        criteria = acs.get("criteria_fired") or []

        print(f"Диагностическая метка: {label}  (confidence={confidence})")
        print(f"Обоснование: {rationale}")
        print(f"Критерии КР: {criteria}")
        print(f"Протокол: {(result.get('protocol') or {}).get('name')}")

        node_trace = result.get("node_trace") or []
        rule_fires = result.get("rule_fires") or []
        rule_reasons = result.get("rule_reasons") or []

        visited = [step.get("node") for step in node_trace]
        print(f"Маршрут по графу: {visited}")
        print(f"Сработало правил: {len(rule_fires)}")
        for fire in rule_fires:
            print(
                f"  - [{fire.get('severity')}] {fire.get('rule_id')}: "
                f"{fire.get('title_ru')}"
            )
        if rule_reasons:
            print(f"Текстовые причины правил: {rule_reasons[:5]}")

        # Проверки
        ok = True
        expected_label = scenario.get("expected_label")
        expected_any = scenario.get("expected_label_any")
        if expected_label and label != expected_label:
            print(
                f"FAIL: ожидался acs_diagnosis.label='{expected_label}', получен '{label}'"
            )
            ok = False
        if expected_any and label not in expected_any:
            print(
                f"FAIL: ожидался acs_diagnosis.label из {expected_any}, получен '{label}'"
            )
            ok = False
        if not node_trace:
            print("FAIL: node_trace пуст - трасса графа не записана.")
            ok = False
        if "classify_acs" not in visited:
            print("FAIL: узел classify_acs не посещён.")
            ok = False
        if ok:
            print("PASS")
        else:
            all_ok = False

    print("\n" + "=" * 78)
    print("ИТОГ:", "OK" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
