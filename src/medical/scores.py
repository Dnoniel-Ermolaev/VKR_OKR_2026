from __future__ import annotations

from typing import Dict


def heart_score(patient_data: Dict[str, object]) -> int:
    score = 0
    if str(patient_data.get("pain_type", "")).lower() == "typical":
        score += 2
    ecg = str(patient_data.get("ecg_changes", "")).lower()
    if "st-depression" in ecg:
        score += 2
    elif ecg not in ("normal", ""):
        score += 1
    troponin = float(patient_data.get("troponin", 0.0))
    if troponin > 0.2:
        score += 2
    elif troponin > 0.05:
        score += 1
    return min(score, 10)


def grace_score(patient_data: Dict[str, object]) -> int:
    hr = int(patient_data.get("hr", 0))
    troponin = float(patient_data.get("troponin", 0.0))
    base = 60
    base += min(hr, 200) // 4
    base += int(troponin * 100)
    if "st-elevation" in str(patient_data.get("ecg_changes", "")).lower():
        base += 30
    return min(base, 250)
