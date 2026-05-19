from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from src.core.state import AgentState
from src.core.tools import LlmClient, build_repository, build_retriever
from src.infrastructure.db.models import PatientData, PatientRecord
from src.medical.diagnosis import AcsDiagnosis, classify_acs as classify_acs_func
from src.medical.rules import evaluate_hard_rules, evaluate_rules, fires_to_jsonable
from src.medical.terminology import diagnosis_label


BASE_DIR = Path(__file__).resolve().parents[2]
REPOSITORY = build_repository(BASE_DIR)
RETRIEVER = build_retriever(BASE_DIR)
LLM_CLIENT = LlmClient()

# Минимум для rule-based скрининга ОКС (синхронно с data_payload_builder.REQUIRED_FIELDS).


def _has_minimal_screening_data(patient_data: Dict[str, object]) -> bool:
    """True, если заполнен структурированный набор для ``rule_check``/``classify_acs``.

    SpO2, ЧДД, глюкоза и прочее - желательны в карте, но не блокируют pretriage:
    иначе LLM по free-form промпту засылает пользователя в ``data_quality_issue``,
    хотя ЭКГ+тропонин+гемодинамика уже есть.
    """
    try:
        name = str(patient_data.get("name", "")).strip()
        pain = patient_data.get("pain_type")
        ecg = str(patient_data.get("ecg_changes", "")).strip()
        trop = patient_data.get("troponin")
        hr = patient_data.get("hr")
        bp = str(patient_data.get("bp", "")).strip()
        if not name or not ecg or not bp:
            return False
        if pain not in ("typical", "atypical", "none"):
            return False
        if trop is None:
            return False
        float(trop)  # тропонин должен быть числом
        if hr is None or int(hr) <= 0:
            return False
        parts = bp.split("/")
        if len(parts) != 2:
            return False
        int(parts[0].strip())
        int(parts[1].strip())
    except (TypeError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# Трасса по графу: каждый узел в начале выполнения логирует своё имя
# и краткий снимок state. Используется фронтендом для отрисовки маршрута
# пациента поверх канонического графа.
# ---------------------------------------------------------------------------
TRACE_SNAPSHOT_KEYS = (
    "next_step",
    "triage_category",
    "risk",
    "risk_level",
    "route_confidence",
    "route_reason",
)


def _trace_entry(node: str, state: AgentState, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Сформировать одну запись трассы.

    Не модифицирует state: возвращает чистый dict, который вызывающая функция
    должна положить обратно в state через ключ ``node_trace``.
    """
    snapshot: Dict[str, Any] = {}
    for key in TRACE_SNAPSHOT_KEYS:
        if key in state:
            snapshot[key] = state.get(key)
    entry: Dict[str, Any] = {
        "node": node,
        "ts": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
    }
    if extra:
        entry.update(extra)
    return entry


def _append_trace(state: AgentState, node: str, extra: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Вернуть обновлённый список ``node_trace`` с новой записью узла."""
    trace = list(state.get("node_trace", []) or [])
    trace.append(_trace_entry(node, state, extra=extra))
    return trace


def llm_parse_history(state: AgentState) -> AgentState:
    trace = _append_trace(state, "llm_parse_history")
    free_text = str(state.get("free_text", "")).strip()
    if not free_text:
        return {
            "missing_fields": ["free_text"],
            "parse_confidence": 0.0,
            "llm_used": False,
            "node_trace": trace,
        }
    patient_data, missing_fields, confidence, llm_used = LLM_CLIENT.parse_history(
        free_text=free_text,
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    return {
        "patient_data": patient_data,
        "missing_fields": missing_fields,
        "parse_confidence": confidence,
        "llm_used": llm_used,
        "node_trace": trace,
    }


def _safe_patient_payload(payload: Dict[str, object]) -> Dict[str, object]:
    safe = dict(payload)
    safe.setdefault("name", "Unknown")
    safe.setdefault("pain_type", "none")
    safe.setdefault("ecg_changes", "normal")
    safe.setdefault("troponin", 0.0)
    safe.setdefault("hr", 70)
    safe.setdefault("bp", "120/80")
    safe.setdefault("symptoms_text", str(payload.get("symptoms_text", "")))
    safe.setdefault("age", None)
    safe.setdefault("gender", "unknown")
    safe.setdefault("admission_time", "")
    safe.setdefault("pain_onset_time", "")
    safe.setdefault("pain_description", str(payload.get("symptoms_text", "")))
    safe.setdefault("spo2", None)
    safe.setdefault("rr", None)
    safe.setdefault("glucose", None)
    safe.setdefault("creatinine", None)
    safe.setdefault("ast_alt_ckmb", {})
    safe.setdefault("lipid_profile", {})
    safe.setdefault("potassium_sodium_magnesium", {})
    safe.setdefault("echo_dkg_results", "")
    safe.setdefault("mri_results", "")
    safe.setdefault("ct_coronary", "")
    safe.setdefault("killip_class", "")
    safe.setdefault("interventions", [])
    safe.setdefault("medications", [])
    safe.setdefault("vital_signs", [])
    return safe


def parse_input(state: AgentState) -> AgentState:
    trace = _append_trace(state, "parse_input")
    payload = state.get("patient_data", {})
    parse_confidence = float(state.get("parse_confidence", 1.0))
    missing_fields = list(state.get("missing_fields", []))
    parsed_ok = True
    try:
        validated = PatientData(**payload)
        normalized = validated.model_dump()
    except ValidationError as exc:
        parsed_ok = False
        normalized = _safe_patient_payload(payload)
        for err in exc.errors():
            loc = err.get("loc", [])
            if loc:
                missing_fields.append(str(loc[0]))
        parse_confidence = min(parse_confidence, 0.35)
    missing_fields = sorted(set(missing_fields))
    return {
        "patient_data": normalized,
        "parsed_ok": parsed_ok,
        "free_text": str(state.get("free_text", "")),
        "missing_fields": missing_fields,
        "parse_confidence": parse_confidence,
        "clarification_attempts": int(state.get("clarification_attempts", 0)),
        "max_clarification_attempts": int(state.get("max_clarification_attempts", 2)),
        "require_llm": bool(state.get("require_llm", False)),
        "force_llm": bool(state.get("force_llm", False)),
        "llm_model": str(state.get("llm_model", "")).strip(),
        "node_trace": trace,
    }


def router_pretriage(state: AgentState) -> AgentState:
    trace = _append_trace(state, "router_pretriage")
    missing_fields = list(state.get("missing_fields", []))
    parsed_ok = bool(state.get("parsed_ok", False))
    patient_data = state["patient_data"]

    # Не отдаём полностью заполненную форму на усмотрение LLM: модель часто просит
    # SpO2/ЧДД/ЭхоКГ и блокирует ``rule_check``, хотя по дизайну прототипа
    # достаточно: ФИО, тип боли, ЭКГ, тропонин, ЧСС, АД.
    if parsed_ok and _has_minimal_screening_data(patient_data):
        return {
            "next_step": "proceed",
            "route_confidence": 0.99,
            "route_reason": (
                "Заполнен минимальный структурированный набор для скрининга ОКС по КР "
                "(тип боли, ЭКГ, тропонин, ЧСС, АД). Дополнительные параметры "
                "(SpO2, ЧДД, лаборатория расширенная) не обязательны для rule-based этапа."
            ),
            "llm_used": False,
            "node_trace": trace,
        }

    next_step, confidence, reason, llm_used = LLM_CLIENT.route_pretriage(
        patient_data=patient_data,
        parse_confidence=float(state.get("parse_confidence", 1.0)),
        missing_fields=missing_fields,
        parsed_ok=parsed_ok,
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    # Confidence gates.
    if confidence < 0.55:
        critical = {"troponin", "ecg_changes", "hr", "bp"}
        if critical.intersection(set(missing_fields)) or not bool(state.get("parsed_ok", False)):
            next_step = "needs_more_data"
            reason = "Низкая уверенность pretriage + не хватает критичных полей."
        else:
            next_step = "proceed"
            reason = "Низкая уверенность pretriage, но минимально достаточные данные."
    return {
        "next_step": next_step,
        "route_confidence": confidence,
        "route_reason": reason,
        "llm_used": llm_used,
        "node_trace": trace,
    }


def clarify_data(state: AgentState) -> AgentState:
    trace = _append_trace(state, "clarify_data")
    attempts = int(state.get("clarification_attempts", 0)) + 1
    max_attempts = int(state.get("max_clarification_attempts", 2))
    missing = state.get("missing_fields", [])
    if attempts <= max_attempts and str(state.get("free_text", "")).strip():
        return {
            "clarification_attempts": attempts,
            "next_step": "retry_parse",
            "route_reason": f"Уточнение попытка {attempts}/{max_attempts}; missing={', '.join(missing)}",
            "node_trace": trace,
        }
    explanation = (
        "Качество данных недостаточно для безопасного клинического решения. "
        f"Не хватает ключевых полей: {', '.join(missing) if missing else 'неизвестно'}. "
        "Требуется ручное уточнение анамнеза / параметров ЭКГ и тропонина."
    )
    return {
        "clarification_attempts": attempts,
        "next_step": "data_quality_issue",
        "triage_category": "data_quality_issue",
        "risk": float(state.get("risk", 0.2)),
        "risk_level": str(state.get("risk_level", "low")),
        "explanation": explanation,
        "node_trace": trace,
    }


def rule_check(state: AgentState) -> AgentState:
    trace = _append_trace(state, "rule_check")
    risk, risk_level, reasons, route_to_llm = evaluate_hard_rules(state["patient_data"])
    fires = evaluate_rules(dict(state["patient_data"]))
    if bool(state.get("force_llm")):
        route_to_llm = True
    return {
        "risk": risk,
        "risk_level": risk_level,
        "rule_reasons": reasons,
        "rule_fires": fires_to_jsonable(fires),
        "route_to_llm": route_to_llm,
        "node_trace": trace,
    }


def classify_acs_node(state: AgentState) -> AgentState:
    """Детерминированная классификация ОКС/ИМ по критериям Минздрав КР.

    Узел выполняется сразу после ``rule_check`` и не зависит от LLM.
    Эмитирует поле ``acs_diagnosis`` - структурированную метку с обоснованием.
    """
    trace = _append_trace(state, "classify_acs")
    rule_reasons = list(state.get("rule_reasons", []))
    diagnosis = classify_acs_func(state["patient_data"], rule_reasons=rule_reasons)

    # Если правила выставили диагностическую метку (sets_diagnosis), приоритизируем
    # её среди срабатываний.
    fired_diag_codes = [
        fire.get("sets_diagnosis")
        for fire in (state.get("rule_fires") or [])
        if fire.get("sets_diagnosis")
    ]
    if fired_diag_codes:
        from src.medical.diagnosis import merge_diagnoses
        candidates: List[AcsDiagnosis] = []
        for code in fired_diag_codes:
            try:
                candidates.append(AcsDiagnosis(code))
            except ValueError:
                continue
        candidates.append(diagnosis.label)
        merged = merge_diagnoses(*candidates)
        if merged != diagnosis.label:
            # Если приоритет правил даёт более тяжёлую метку - переписываем.
            diagnosis.label = merged
            diagnosis.icd10_suggested = list(
                {*diagnosis.icd10_suggested, *_icd_suggestions_for(merged)}
            )

    return {
        "acs_diagnosis": diagnosis.to_dict(),
        "node_trace": trace,
    }


def _icd_suggestions_for(label: AcsDiagnosis) -> List[str]:
    from src.medical.diagnosis import ICD_SUGGESTIONS
    return list(ICD_SUGGESTIONS.get(label, []))


def router_diagnostic(state: AgentState) -> AgentState:
    trace = _append_trace(state, "router_diagnostic")
    next_step, confidence, reason, llm_used = LLM_CLIENT.route_diagnostic(
        patient_data=state["patient_data"],
        rule_reasons=state.get("rule_reasons", []),
        risk=float(state.get("risk", 0.0)),
        risk_level=str(state.get("risk_level", "low")),
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    if confidence < 0.6:
        risk = float(state.get("risk", 0.0))
        if risk >= 0.9:
            next_step = "urgent"
            reason = "Низкая уверенность diagnostic-роутера; эскалируем по risk."
        elif risk >= 0.45:
            next_step = "rag_llm"
            reason = "Низкая уверенность diagnostic-роутера; неопределённый маршрут."
        else:
            next_step = "rule_only"
            reason = "Низкая уверенность diagnostic-роутера; маршрут наблюдения."
    if bool(state.get("force_llm")) and next_step == "rule_only":
        next_step = "rag_llm"
        reason = "force_llm активен, переход в diagnostic_uncertain."
    return {
        "next_step": next_step,
        "route_confidence": confidence,
        "route_reason": reason,
        "llm_used": llm_used,
        "node_trace": trace,
    }


def high_risk_fast_track(state: AgentState) -> AgentState:
    trace = _append_trace(state, "high_risk_fast_track")
    diag_code = _diagnosis_code(state)
    base = (
        "Маршрут high_risk_fast_track: высокий риск по правилам клинических "
        "рекомендаций Минздрава, требуется срочная кардиологическая тактика "
        "(оценка показаний к КГ и ЧКВ, ДАТТ, антикоагулянт)."
    )
    if diag_code in {"im_pst", "oks_pst"}:
        base += " Приоритет - первичная реперфузия (ЧКВ <=120 мин)."
    elif diag_code in {"im_bpst", "oks_bpst"}:
        base += " Инвазивная стратегия в сроки по GRACE-риску."
    return {
        "triage_category": "high_risk_fast_track",
        "explanation": base,
        "node_trace": trace,
    }


def diagnostic_uncertain(state: AgentState) -> AgentState:
    trace = _append_trace(state, "diagnostic_uncertain")
    return {"triage_category": "diagnostic_uncertain", "node_trace": trace}


def low_risk_observation(state: AgentState) -> AgentState:
    trace = _append_trace(state, "low_risk_observation")
    explanation = state.get("explanation") or (
        "Маршрут low_risk_observation: динамическое наблюдение, серия тропонинов и ЭКГ."
    )
    return {
        "triage_category": "low_risk_observation",
        "explanation": explanation,
        "node_trace": trace,
    }


def data_quality_issue(state: AgentState) -> AgentState:
    trace = _append_trace(state, "data_quality_issue")
    return {
        "triage_category": "data_quality_issue",
        "risk": float(state.get("risk", 0.2)),
        "risk_level": str(state.get("risk_level", "low")),
        "explanation": state.get("explanation") or (
            "Недостаточно данных для безопасного клинического решения по ОКС."
        ),
        "node_trace": trace,
    }


def _build_rag_query(patient: Dict[str, object]) -> str:
    fields = [
        "клинические рекомендации ОКС ИМпST ИМбпST триаж",
        f"pain_type: {patient.get('pain_type', '')}",
        f"ecg_changes: {patient.get('ecg_changes', '')}",
        f"troponin: {patient.get('troponin', '')}",
        f"hr: {patient.get('hr', '')}",
        f"bp: {patient.get('bp', '')}",
        f"spo2: {patient.get('spo2', '')}",
        f"creatinine: {patient.get('creatinine', '')}",
        f"killip_class: {patient.get('killip_class', '')}",
        f"age: {patient.get('age', '')}",
        f"gender: {patient.get('gender', '')}",
        f"symptoms: {patient.get('symptoms_text', '')}",
    ]
    return "\n".join(str(item) for item in fields if str(item).strip())


def rag_retrieval(state: AgentState) -> AgentState:
    trace = _append_trace(state, "rag_retrieval")
    patient = state["patient_data"]
    query = _build_rag_query(patient)
    hits = RETRIEVER.retrieve_hits(query=query, top_k=3)
    snippets = [hit.formatted_text(rank=idx + 1) for idx, hit in enumerate(hits)]
    citations = [hit.citation for hit in hits]
    return {
        "rag_context": "\n---\n".join(snippets),
        "citations": citations,
        "node_trace": trace,
    }


def llm_assess(state: AgentState) -> AgentState:
    trace = _append_trace(state, "llm_assess")
    adjustment, explanation, llm_used = LLM_CLIENT.assess(
        patient_data=state["patient_data"],
        rule_reasons=state.get("rule_reasons", []),
        rag_context=state.get("rag_context", ""),
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    new_risk = max(0.0, min(1.0, float(state.get("risk", 0.0)) + adjustment))
    if new_risk >= 0.75:
        level = "high"
    elif new_risk >= 0.45:
        level = "medium"
    else:
        level = "low"
    citations = list(state.get("citations", []))
    if citations:
        explanation = f"{explanation} Источники: {'; '.join(citations[:3])}."
    return {
        "risk": new_risk,
        "risk_level": level,
        "explanation": explanation,
        "llm_used": llm_used,
        "triage_category": "diagnostic_uncertain",
        "citations": citations,
        "node_trace": trace,
    }


def router_management(state: AgentState) -> AgentState:
    trace = _append_trace(state, "router_management")
    next_step, confidence, reason, llm_used = LLM_CLIENT.route_management(
        patient_data=state["patient_data"],
        risk=float(state.get("risk", 0.0)),
        risk_level=str(state.get("risk_level", "low")),
        explanation=str(state.get("explanation", "")),
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    if confidence < 0.55:
        if str(state.get("risk_level", "low")) == "high":
            next_step = "recommend_treatment"
            reason = "Низкая уверенность management-роутера; при высоком риске - рекомендации."
        else:
            next_step = "monitor"
            reason = "Низкая уверенность management-роутера; маршрут мониторинга."
    return {
        "next_step": next_step,
        "route_confidence": confidence,
        "route_reason": reason,
        "llm_used": llm_used,
        "node_trace": trace,
    }


def monitor_plan(state: AgentState) -> AgentState:
    trace = _append_trace(state, "monitor_plan")
    append_text = (
        "Рекомендован мониторинг: серийная ЭКГ, АД, ЧСС, SpO2 в динамике "
        "и повторный тропонин (0/1 ч или 0/3 ч)."
    )
    explanation = f"{state.get('explanation', '')} {append_text}".strip()
    return {"explanation": explanation, "node_trace": trace}


def recommend_treatment(state: AgentState) -> AgentState:
    trace = _append_trace(state, "recommend_treatment")
    risk_level = str(state.get("risk_level", "low"))
    diag_code = _diagnosis_code(state)
    if risk_level == "high":
        plan_parts = [
            "Предварительный план:",
            "- срочная кардиологическая тактика (оценка показаний к КГ/ЧКВ);",
            "- ДАТТ (АСК + ингибитор P2Y12), антикоагулянт по протоколу;",
            "- контроль гемодинамики, по показаниям - оксигенотерапия;",
        ]
        if diag_code in {"im_pst", "oks_pst"}:
            plan_parts.append(
                "- приоритет - первичное ЧКВ в пределах 120 минут "
                "от первого медицинского контакта."
            )
        elif diag_code in {"im_bpst", "oks_bpst"}:
            plan_parts.append(
                "- инвазивная стратегия в сроки по GRACE-риску "
                "(<=2 ч очень высокий риск, <=24 ч высокий риск)."
            )
        plan = " ".join(plan_parts)
    else:
        plan = (
            "Предварительный план: динамическое наблюдение, серия тропонинов и ЭКГ, "
            "контроль симптомов; при сохранении неопределённости - рассмотреть "
            "стресс-тест или КТ-коронарографию."
        )
    explanation = f"{state.get('explanation', '')} {plan}".strip()
    return {"explanation": explanation, "node_trace": trace}


def output_save(state: AgentState) -> AgentState:
    trace = _append_trace(state, "output_save")
    explanation = state.get("explanation") or "Оценка выполнена по rule-based логике (КР Минздрав)."
    diag_code = _diagnosis_code(state)
    if diag_code and diag_code != "oks_unlikely":
        explanation = f"Диагностическая метка: {diagnosis_label(diag_code)}. {explanation}".strip()
    record_id = str(uuid.uuid4())
    record = PatientRecord.from_assessment(
        record_id=record_id,
        patient_data=state["patient_data"],
        risk_level=state["risk_level"],
        explanation=explanation,
    )
    REPOSITORY.save_patient(record)
    return {"save_id": record_id, "explanation": explanation, "node_trace": trace}


def _diagnosis_code(state: AgentState) -> str:
    raw = state.get("acs_diagnosis")
    if isinstance(raw, dict):
        return str(raw.get("label", "") or "")
    return ""


def route_from_start(state: AgentState) -> str:
    return "llm_parse_history" if bool(str(state.get("free_text", "")).strip()) else "parse_input"


def route_after_pretriage(state: AgentState) -> str:
    step = str(state.get("next_step", "proceed"))
    return "clarify_data" if step == "needs_more_data" else "rule_check"


def route_after_clarify(state: AgentState) -> str:
    step = str(state.get("next_step", "data_quality_issue"))
    if step == "retry_parse":
        return "llm_parse_history"
    return "data_quality_issue"


def route_after_diagnostic(state: AgentState) -> str:
    step = str(state.get("next_step", "rag_llm"))
    if step == "urgent":
        return "high_risk_fast_track"
    if step == "rule_only":
        return "low_risk_observation"
    return "diagnostic_uncertain"


def route_after_management(state: AgentState) -> str:
    step = str(state.get("next_step", "finalize"))
    if step == "monitor":
        return "monitor_plan"
    if step == "recommend_treatment":
        return "recommend_treatment"
    return "output_save"


# ``classify_acs`` - публичное имя узла графа, чтобы не конфликтовать с
# функцией ``classify_acs`` из ``src.medical.diagnosis``.
classify_acs = classify_acs_node
