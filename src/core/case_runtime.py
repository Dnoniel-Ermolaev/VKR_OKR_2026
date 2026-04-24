from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List


CRITICAL_FIELDS = {"troponin", "ecg_changes", "hr", "bp"}
NUMERIC_FIELDS = {"troponin", "hr", "spo2", "rr", "glucose", "creatinine", "age"}
TEXT_FIELDS = {"bp", "ecg_changes", "killip_class", "symptoms_text", "pain_type", "gender", "echo_dkg_results"}


def infer_case_status(result: Dict[str, Any], protocol_pending: bool = False) -> tuple[str, str]:
    """Определить статус и стадию кейса.

    protocol_pending=True означает, что по клиническому протоколу ещё остались
    обязательные шаги (серийный тропонин, ЭхоКГ и т. п.), поэтому кейс нельзя
    считать завершённым автоматически.
    """
    missing = set(result.get("missing_fields", []))
    triage_category = str(result.get("triage_category", ""))
    next_step = str(result.get("next_step", ""))
    if triage_category == "data_quality_issue" or missing.intersection(CRITICAL_FIELDS):
        return "awaiting_labs", "awaiting_labs"
    if protocol_pending:
        return "active", "protocol_in_progress"
    if next_step == "monitor":
        return "active", "monitoring"
    return "active", triage_category or "active"


def normalize_observations(observations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in observations:
        code = str(item.get("code", "")).strip()
        name = str(item.get("name", "")).strip() or code
        if not name and not code:
            continue
        category = str(item.get("category", "vital")).strip() or "vital"
        unit = str(item.get("unit", "")).strip()
        value_num = item.get("value_num")
        value_text = item.get("value_text")
        if value_num is None and item.get("value") is not None:
            raw_value = item.get("value")
            if isinstance(raw_value, (int, float)):
                value_num = float(raw_value)
            else:
                value_text = str(raw_value)
        normalized.append(
            {
                "category": category,
                "code": code,
                "name": name,
                "value_num": float(value_num) if value_num is not None else None,
                "value_text": str(value_text) if value_text is not None else None,
                "unit": unit,
                "flag": str(item.get("flag", "unknown")),
                "source": str(item.get("source", "manual")),
                "note": str(item.get("note", "")),
                "recorded_at": item.get("recorded_at"),
            }
        )
    return normalized


LAB_CODE_TO_FIELD = {
    "troponin_i": "troponin",
    "troponin_t": "troponin",
    "creatinine_blood": "creatinine",
    "glucose_blood": "glucose",
}
LAB_CODE_TO_DICT_FIELD = {
    "ast_blood": ("ast_alt_ckmb", "ast"),
    "alt_blood": ("ast_alt_ckmb", "alt"),
    "ck_mb": ("ast_alt_ckmb", "ckmb"),
    "cholesterol_total": ("lipid_profile", "total"),
    "ldl": ("lipid_profile", "ldl"),
    "hdl": ("lipid_profile", "hdl"),
    "vldl": ("lipid_profile", "vldl"),
    "triglycerides": ("lipid_profile", "triglycerides"),
    "atherogenic_index": ("lipid_profile", "atherogenic_index"),
    "k_blood": ("potassium_sodium_magnesium", "k"),
    "na_blood": ("potassium_sodium_magnesium", "na"),
    "mg_blood": ("potassium_sodium_magnesium", "mg"),
}
VITAL_CODE_TO_FIELD = {
    "hr": "hr",
    "rr": "rr",
    "spo2": "spo2",
    "temp": "temperature",
    "glucose_bedside": "glucose",
}


def merge_payload_with_observations(
    payload: Dict[str, Any], observations: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    """Обновить `patient_data`-payload последними значениями наблюдений.

    Учитывается и старый формат (name без code), и новый (code из каталога).
    """
    merged = dict(payload)
    vital_signs = list(merged.get("vital_signs", []))

    sbp_latest: float | None = None
    dbp_latest: float | None = None
    ast_bucket: Dict[str, float] = dict(merged.get("ast_alt_ckmb", {}) or {})
    lipid_bucket: Dict[str, float] = dict(merged.get("lipid_profile", {}) or {})
    electro_bucket: Dict[str, float] = dict(merged.get("potassium_sodium_magnesium", {}) or {})

    for obs in observations:
        code = str(obs.get("code", "") or "").strip().lower()
        name = str(obs.get("name", "")).strip()
        category = str(obs.get("category", "vital"))
        value_num = obs.get("value_num")
        value_text = obs.get("value_text")

        # Map by catalog code first.
        if code in LAB_CODE_TO_FIELD and value_num is not None:
            field = LAB_CODE_TO_FIELD[code]
            merged[field] = float(value_num)
        elif code in LAB_CODE_TO_DICT_FIELD and value_num is not None:
            bucket_name, key = LAB_CODE_TO_DICT_FIELD[code]
            if bucket_name == "ast_alt_ckmb":
                ast_bucket[key] = float(value_num)
            elif bucket_name == "lipid_profile":
                lipid_bucket[key] = float(value_num)
            elif bucket_name == "potassium_sodium_magnesium":
                electro_bucket[key] = float(value_num)
        elif code == "sbp" and value_num is not None:
            sbp_latest = float(value_num)
        elif code == "dbp" and value_num is not None:
            dbp_latest = float(value_num)
        elif code in VITAL_CODE_TO_FIELD and value_num is not None:
            field = VITAL_CODE_TO_FIELD[code]
            value = float(value_num)
            if field == "hr" or field == "rr":
                value = int(round(value))
            merged[field] = value
        else:
            # Legacy path: fall back on `name` being a raw PatientData field.
            if not name:
                continue
            if value_num is not None:
                value: Any = float(value_num)
                if name in {"hr", "rr", "age"}:
                    value = int(round(value))
                merged[name] = value
            elif value_text is not None:
                merged[name] = str(value_text)

        if category == "vital":
            vital_signs.append(
                {
                    "code": code,
                    "name": name,
                    "value_num": value_num,
                    "value_text": value_text,
                    "unit": obs.get("unit", ""),
                    "recorded_at": obs.get("recorded_at"),
                }
            )

    if sbp_latest is not None or dbp_latest is not None:
        existing_bp = str(merged.get("bp", "120/80"))
        if "/" in existing_bp:
            try:
                cur_sbp_str, cur_dbp_str = existing_bp.split("/", 1)
                cur_sbp = int(float(cur_sbp_str.strip())) if cur_sbp_str.strip() else 120
                cur_dbp = int(float(cur_dbp_str.strip())) if cur_dbp_str.strip() else 80
            except ValueError:
                cur_sbp, cur_dbp = 120, 80
        else:
            cur_sbp, cur_dbp = 120, 80
        new_sbp = int(round(sbp_latest)) if sbp_latest is not None else cur_sbp
        new_dbp = int(round(dbp_latest)) if dbp_latest is not None else cur_dbp
        merged["bp"] = f"{new_sbp}/{new_dbp}"

    if ast_bucket:
        merged["ast_alt_ckmb"] = ast_bucket
    if lipid_bucket:
        merged["lipid_profile"] = lipid_bucket
    if electro_bucket:
        merged["potassium_sodium_magnesium"] = electro_bucket
    if vital_signs:
        merged["vital_signs"] = vital_signs
    return merged


def merge_payload_with_case(
    payload: Dict[str, Any],
    observations: Iterable[Dict[str, Any]] | None = None,
    medications: Iterable[Any] | None = None,
    diagnoses: Iterable[Any] | None = None,
    studies: Iterable[Any] | None = None,
    procedures: Iterable[Any] | None = None,
) -> Dict[str, Any]:
    """Полный merge: наблюдения + активные медикаменты + диагнозы + исследования."""
    obs_list = list(observations or [])
    merged = merge_payload_with_observations(payload, obs_list)

    def _field(item: Any, key: str) -> Any:
        return getattr(item, key, None) if hasattr(item, key) else item.get(key)

    if medications is not None:
        med_names = [str(_field(m, "name") or "").strip() for m in medications if _field(m, "name")]
        if med_names:
            merged.setdefault("medications", []).extend(
                [n for n in med_names if n not in merged.get("medications", [])]
            )

    if diagnoses is not None:
        primary_names = [
            str(_field(d, "name") or "").strip()
            for d in diagnoses
            if _field(d, "name") and (_field(d, "diagnosis_type") or "primary") == "primary"
        ]
        if primary_names and not merged.get("symptoms_text"):
            merged["symptoms_text"] = "; ".join(primary_names)

    if studies is not None or procedures is not None:
        names: List[str] = []
        for item in list(studies or []) + list(procedures or []):
            item_name = str(_field(item, "name") or "").strip()
            status = str(_field(item, "status") or "").lower()
            if item_name and status in {"done", "completed"}:
                names.append(item_name)
        if names:
            merged.setdefault("interventions", []).extend(
                [n for n in names if n not in merged.get("interventions", [])]
            )

    return merged


def build_series(observations: Iterable[Any]) -> Dict[str, List[Dict[str, Any]]]:
    series: Dict[str, List[Dict[str, Any]]] = {}
    for obs in observations:
        is_obj = hasattr(obs, "__dict__")
        name = getattr(obs, "name", None) if is_obj else obs.get("name")
        if not name:
            continue
        item = {
            "category": getattr(obs, "category", None) if is_obj else obs.get("category"),
            "value_num": getattr(obs, "value_num", None) if is_obj else obs.get("value_num"),
            "value_text": getattr(obs, "value_text", None) if is_obj else obs.get("value_text"),
            "unit": (getattr(obs, "unit", None) if is_obj else obs.get("unit", "")) or "",
            "recorded_at": _iso(getattr(obs, "recorded_at", None) if is_obj else obs.get("recorded_at")),
            "source": (getattr(obs, "source", None) if is_obj else obs.get("source", "")) or "",
        }
        series.setdefault(str(name), []).append(item)
    return series


def build_case_title(payload: Dict[str, Any]) -> str:
    name = str(payload.get("name", "Unknown")).strip() or "Unknown"
    symptoms = str(payload.get("symptoms_text", "")).strip()
    if symptoms:
        return f"{name}: {symptoms[:60]}"
    return f"{name}: ACS case"


def make_report_query(case_payload: Dict[str, Any], latest_result: Dict[str, Any], series: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = [
        "clinical ACS report and epicrisis",
        f"name: {case_payload.get('name', '')}",
        f"pain_type: {case_payload.get('pain_type', '')}",
        f"ecg_changes: {case_payload.get('ecg_changes', '')}",
        f"troponin: {case_payload.get('troponin', '')}",
        f"risk_level: {latest_result.get('risk_level', '')}",
        f"triage_category: {latest_result.get('triage_category', '')}",
        f"explanation: {latest_result.get('explanation', '')}",
    ]
    if series:
        for key, values in series.items():
            tail = values[-3:]
            rendered = ", ".join(
                str(item.get("value_num") if item.get("value_num") is not None else item.get("value_text"))
                for item in tail
            )
            lines.append(f"time_series {key}: {rendered}")
    return "\n".join(lines)


def build_case_summary(case_payload: Dict[str, Any], latest_result: Dict[str, Any], series: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = [
        f"Пациент: {case_payload.get('name', 'Unknown')}",
        f"Возраст: {case_payload.get('age', 'неизвестно')}, пол: {case_payload.get('gender', 'unknown')}",
        f"Симптомы: {case_payload.get('symptoms_text', '')}",
        f"Боль: {case_payload.get('pain_type', '')}, ЭКГ: {case_payload.get('ecg_changes', '')}",
        f"Тропонин: {case_payload.get('troponin', '')}, ЧСС: {case_payload.get('hr', '')}, АД: {case_payload.get('bp', '')}",
        f"Риск: {latest_result.get('risk_level', '')}, маршрут: {latest_result.get('triage_category', '')}",
        f"Объяснение: {latest_result.get('explanation', '')}",
    ]
    if series:
        lines.append("Динамика наблюдений:")
        for name, items in sorted(series.items()):
            tail = items[-5:]
            rendered = "; ".join(
                f"{item.get('recorded_at')}: {item.get('value_num') if item.get('value_num') is not None else item.get('value_text')}{item.get('unit', '')}"
                for item in tail
            )
            lines.append(f"- {name}: {rendered}")
    return "\n".join(lines)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
