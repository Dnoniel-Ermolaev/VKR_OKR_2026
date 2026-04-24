"""Протоколы клинического контроля ОКС (STEMI / NSTEMI / UA / generic).

Каждый протокол задаёт перечень обязательных наблюдений, исследований,
процедур и групп медикаментов с привязкой к окну времени. На их основе
строится дашборд контроля пациента.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from src.medical.catalog import DIAGNOSIS_BY_ICD


RequirementKind = Literal["lab", "study", "procedure", "medication_class", "vital"]
Priority = Literal["critical", "high", "medium", "low"]


@dataclass(frozen=True)
class ProtocolRequirement:
    kind: RequirementKind
    code: str
    title: str
    window_hours: Optional[float] = None
    repeat_interval_hours: Optional[float] = None
    min_occurrences: int = 1
    priority: Priority = "medium"
    med_class: Optional[str] = None
    note: str = ""


@dataclass(frozen=True)
class Protocol:
    code: str
    name: str
    description: str
    requirements: List[ProtocolRequirement] = field(default_factory=list)


STEMI_PROTOCOL = Protocol(
    code="acs_stemi",
    name="ОКС с подъёмом ST (STEMI)",
    description=(
        "Приоритет — реперфузия. Серийный тропонин, серия ЭКГ, ЭхоКГ в первые 24 ч, "
        "коронарография в целевые сроки, двойная антитромбоцитарная терапия, антикоагулянт, "
        "статин, β-блокатор, иАПФ."
    ),
    requirements=[
        ProtocolRequirement("lab", "troponin_i", "Серия тропонина", window_hours=0,
                            repeat_interval_hours=3, min_occurrences=3, priority="critical",
                            note="Повторные измерения 0h / 3h / 6h."),
        ProtocolRequirement("study", "ecg_12", "Серия ЭКГ", window_hours=0,
                            repeat_interval_hours=6, min_occurrences=3, priority="critical",
                            note="Повторная ЭКГ при любой динамике и минимум 3 раза в первые сутки."),
        ProtocolRequirement("study", "echo_cg", "ЭхоКГ", window_hours=24, priority="high",
                            note="Оценка сократимости и осложнений."),
        ProtocolRequirement("study", "coronary_angiography", "Коронарография", window_hours=2,
                            priority="critical",
                            note="В первые 120 минут при STEMI и возможности ЧКВ."),
        ProtocolRequirement("procedure", "pci_stent", "ЧКВ со стентированием",
                            window_hours=2, priority="critical"),
        ProtocolRequirement("lab", "creatinine_blood", "Креатинин", window_hours=6,
                            priority="high", note="До введения контраста."),
        ProtocolRequirement("lab", "k_blood", "Калий", window_hours=6, priority="high"),
        ProtocolRequirement("lab", "hgb", "Гемоглобин", window_hours=6, priority="medium"),
        ProtocolRequirement("lab", "plt", "Тромбоциты", window_hours=6, priority="medium"),
        ProtocolRequirement("lab", "inr", "МНО", window_hours=24, priority="medium"),
        ProtocolRequirement("lab", "ldl", "ЛПНП (липидный спектр)", window_hours=72,
                            priority="medium"),
        ProtocolRequirement("medication_class", "antiplatelet", "Двойная антиагрегантная терапия",
                            priority="critical", med_class="antiplatelet",
                            min_occurrences=2,
                            note="Аспирин + ингибитор P2Y12 (тикагрелор/клопидогрел/прасугрел)."),
        ProtocolRequirement("medication_class", "anticoag", "Антикоагулянт",
                            priority="critical", med_class="anticoag"),
        ProtocolRequirement("medication_class", "statin", "Высокоинтенсивный статин",
                            priority="high", med_class="statin"),
        ProtocolRequirement("medication_class", "beta_blocker", "β-блокатор",
                            priority="high", med_class="beta_blocker"),
        ProtocolRequirement("medication_class", "acei", "иАПФ/БРА",
                            priority="medium", med_class="acei"),
        ProtocolRequirement("vital", "spo2", "Контроль SpO2", priority="high",
                            note="Оксигенотерапия при SpO2 < 90%."),
    ],
)


NSTEMI_PROTOCOL = Protocol(
    code="acs_nstemi",
    name="ОКС без подъёма ST (NSTEMI)",
    description=(
        "GRACE-ориентированная инвазивная тактика. Серийный тропонин (0/1-3 ч), "
        "ЭКГ в динамике, ЭхоКГ, коронарография в сроки по риску, ДАТТ + антикоагулянт, "
        "статин, β-блокатор."
    ),
    requirements=[
        ProtocolRequirement("lab", "troponin_i", "Серия тропонина (0 / 1-3 ч)",
                            window_hours=0, repeat_interval_hours=3, min_occurrences=2,
                            priority="critical"),
        ProtocolRequirement("study", "ecg_12", "Серия ЭКГ", window_hours=0,
                            repeat_interval_hours=6, min_occurrences=2, priority="critical"),
        ProtocolRequirement("study", "echo_cg", "ЭхоКГ", window_hours=24, priority="high"),
        ProtocolRequirement("study", "coronary_angiography", "Коронарография",
                            window_hours=24, priority="high",
                            note="В пределах 24 ч для высокорисковых NSTEMI."),
        ProtocolRequirement("lab", "creatinine_blood", "Креатинин", window_hours=6,
                            priority="high"),
        ProtocolRequirement("lab", "k_blood", "Калий", window_hours=6, priority="high"),
        ProtocolRequirement("lab", "hgb", "Гемоглобин", window_hours=6, priority="medium"),
        ProtocolRequirement("lab", "ldl", "ЛПНП", window_hours=72, priority="medium"),
        ProtocolRequirement("medication_class", "antiplatelet", "Двойная антиагрегантная терапия",
                            priority="critical", med_class="antiplatelet", min_occurrences=2),
        ProtocolRequirement("medication_class", "anticoag", "Антикоагулянт",
                            priority="critical", med_class="anticoag"),
        ProtocolRequirement("medication_class", "statin", "Статин",
                            priority="high", med_class="statin"),
        ProtocolRequirement("medication_class", "beta_blocker", "β-блокатор",
                            priority="high", med_class="beta_blocker"),
        ProtocolRequirement("vital", "spo2", "Контроль SpO2", priority="high"),
    ],
)


UA_PROTOCOL = Protocol(
    code="acs_ua",
    name="Нестабильная стенокардия (UA)",
    description=(
        "Тропонин-отрицательный ОКС. Наблюдение, серия тропонинов, серия ЭКГ, ЭхоКГ, "
        "ДАТТ/антиагрегант, статин; коронарография при подтверждении ишемии."
    ),
    requirements=[
        ProtocolRequirement("lab", "troponin_i", "Серия тропонина",
                            window_hours=0, repeat_interval_hours=3, min_occurrences=2,
                            priority="high"),
        ProtocolRequirement("study", "ecg_12", "Серия ЭКГ",
                            window_hours=0, repeat_interval_hours=6, min_occurrences=2,
                            priority="high"),
        ProtocolRequirement("study", "echo_cg", "ЭхоКГ", window_hours=48, priority="medium"),
        ProtocolRequirement("lab", "creatinine_blood", "Креатинин", window_hours=24,
                            priority="medium"),
        ProtocolRequirement("lab", "ldl", "ЛПНП", window_hours=72, priority="medium"),
        ProtocolRequirement("medication_class", "antiplatelet", "Антиагрегантная терапия",
                            priority="high", med_class="antiplatelet"),
        ProtocolRequirement("medication_class", "statin", "Статин",
                            priority="medium", med_class="statin"),
        ProtocolRequirement("vital", "spo2", "Контроль SpO2", priority="medium"),
    ],
)


GENERIC_ACS_PROTOCOL = Protocol(
    code="acs_generic",
    name="Подозрение на ОКС (дифференциальная диагностика)",
    description="Минимальный объём обследования при подозрении на ОКС до уточнения диагноза.",
    requirements=[
        ProtocolRequirement("lab", "troponin_i", "Тропонин",
                            window_hours=0, repeat_interval_hours=3, min_occurrences=2,
                            priority="high"),
        ProtocolRequirement("study", "ecg_12", "ЭКГ", window_hours=0, priority="high",
                            repeat_interval_hours=6, min_occurrences=2),
        ProtocolRequirement("lab", "creatinine_blood", "Креатинин", window_hours=24,
                            priority="medium"),
        ProtocolRequirement("lab", "hgb", "Гемоглобин", window_hours=24, priority="low"),
        ProtocolRequirement("vital", "spo2", "SpO2", priority="medium"),
    ],
)


PROTOCOLS: Dict[str, Protocol] = {
    STEMI_PROTOCOL.code: STEMI_PROTOCOL,
    NSTEMI_PROTOCOL.code: NSTEMI_PROTOCOL,
    UA_PROTOCOL.code: UA_PROTOCOL,
    GENERIC_ACS_PROTOCOL.code: GENERIC_ACS_PROTOCOL,
}


def _icd_group(icd10: str) -> str:
    definition = DIAGNOSIS_BY_ICD.get(icd10.upper())
    return definition.group if definition else ""


def select_protocol(
    diagnoses_icd: List[str] | None,
    ecg_description: str = "",
    triage_category: str = "",
) -> Protocol:
    """Выбрать клинический протокол по диагнозам МКБ и контексту.

    Приоритет: явный STEMI по диагнозам, затем по описанию ЭКГ (ST-elevation),
    затем NSTEMI/UA, иначе generic.
    """
    icd_codes = [str(code).upper().strip() for code in (diagnoses_icd or []) if code]
    groups = {_icd_group(code) for code in icd_codes}

    if "STEMI" in groups:
        return STEMI_PROTOCOL
    if "NSTEMI" in groups:
        return NSTEMI_PROTOCOL
    if "UA" in groups:
        return UA_PROTOCOL

    ecg = (ecg_description or "").lower()
    if "st-elevation" in ecg or "элевация" in ecg or "подъём st" in ecg or "подъем st" in ecg:
        return STEMI_PROTOCOL
    if "st-depression" in ecg or "депрессия st" in ecg:
        return NSTEMI_PROTOCOL

    triage = (triage_category or "").lower()
    if "high_risk" in triage:
        return NSTEMI_PROTOCOL

    return GENERIC_ACS_PROTOCOL


def protocol_summary(protocol: Protocol) -> dict:
    return {
        "code": protocol.code,
        "name": protocol.name,
        "description": protocol.description,
        "requirements": [
            {
                "kind": r.kind,
                "code": r.code,
                "title": r.title,
                "window_hours": r.window_hours,
                "repeat_interval_hours": r.repeat_interval_hours,
                "min_occurrences": r.min_occurrences,
                "priority": r.priority,
                "med_class": r.med_class,
                "note": r.note,
            }
            for r in protocol.requirements
        ],
    }


__all__ = [
    "ProtocolRequirement", "Protocol",
    "STEMI_PROTOCOL", "NSTEMI_PROTOCOL", "UA_PROTOCOL", "GENERIC_ACS_PROTOCOL",
    "PROTOCOLS", "select_protocol", "protocol_summary",
]
