from __future__ import annotations

from typing import Dict, List, Tuple


def evaluate_hard_rules(patient_data: Dict[str, object]) -> Tuple[float, str, List[str], bool]:
    """
    Returns (risk, risk_level, reasons, route_to_llm).

    route_to_llm=False when high-risk condition is obvious.
    """
    reasons: List[str] = []
    troponin = float(patient_data.get("troponin", 0.0))
    ecg = str(patient_data.get("ecg_changes", "")).lower()
    pain_type = str(patient_data.get("pain_type", "")).lower()
    hr = int(patient_data.get("hr", 0))
    creatinine = patient_data.get("creatinine")
    spo2 = patient_data.get("spo2")
    killip = str(patient_data.get("killip_class", "")).upper()

    if "st-elevation" in ecg:
        reasons.append("ECG indicates ST-elevation (possible STEMI).")
        return 0.98, "high", reasons, False

    if troponin >= 0.5:
        reasons.append("Very high troponin level.")
        return 0.92, "high", reasons, False

    risk = 0.2
    if pain_type == "typical":
        risk += 0.2
        reasons.append("Typical chest pain pattern.")
    if "st-depression" in ecg:
        risk += 0.25
        reasons.append("ECG ST-depression present.")
    if troponin >= 0.1:
        risk += 0.2
        reasons.append("Elevated troponin.")
    if hr > 110:
        risk += 0.1
        reasons.append("Tachycardia > 110 bpm.")
    if isinstance(creatinine, (int, float)) and float(creatinine) >= 140:
        risk += 0.1
        reasons.append("Elevated creatinine suggests higher risk.")
    if isinstance(spo2, (int, float)) and float(spo2) < 90:
        risk += 0.1
        reasons.append("Hypoxemia (SpO2 < 90%).")
    if killip in {"III", "IV", "3", "4"}:
        risk += 0.15
        reasons.append("High Killip class.")

    if risk >= 0.75:
        return min(risk, 0.95), "high", reasons, False
    if risk >= 0.45:
        return risk, "medium", reasons, True
    return risk, "low", reasons, True
