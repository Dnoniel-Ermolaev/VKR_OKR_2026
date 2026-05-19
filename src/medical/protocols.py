"""Клинические протоколы ведения ОКС (ИМпST / ИМбпST / НС / generic).

Названия и описания приведены в соответствие с терминологией клинических
рекомендаций Минздрава РФ 2020 г. (ИМпST, ОКСбпST / ИМбпST,
нестабильная стенокардия - НС).

Каждый протокол задаёт перечень обязательных наблюдений, исследований,
процедур и групп медикаментов с привязкой к окну времени.
На их основе строится дашборд контроля пациента.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from src.medical.catalog import DIAGNOSIS_BY_ICD
from src.medical.diagnosis import AcsDiagnosis


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
    name="ОКСпST / ИМпST - острый коронарный синдром с подъёмом сегмента ST",
    description=(
        "Приоритет - первичная реперфузия (первичное ЧКВ <=120 мин при возможности). "
        "Серия тропонина 0/3/6 ч, ЭКГ в динамике, ЭхоКГ в первые 24 ч, коронарография и ЧКВ "
        "со стентированием в целевые сроки. ДАТТ (АСК + ингибитор P2Y12), антикоагулянт, "
        "высокоинтенсивный статин, бета-блокатор, ИАПФ/БРА."
    ),
    requirements=[
        ProtocolRequirement("lab", "troponin_i", "Серия тропонина", window_hours=0,
                            repeat_interval_hours=3, min_occurrences=3, priority="critical",
                            note="Повторные измерения 0 / 3 / 6 ч."),
        ProtocolRequirement("study", "ecg_12", "Серия ЭКГ 12 отведений", window_hours=0,
                            repeat_interval_hours=6, min_occurrences=3, priority="critical",
                            note="Повторная ЭКГ при любой динамике, минимум 3 раза в первые сутки."),
        ProtocolRequirement("study", "echo_cg", "ЭхоКГ", window_hours=24, priority="high",
                            note="Оценка сократимости и осложнений."),
        ProtocolRequirement("study", "coronary_angiography", "Коронарография (КГ)",
                            window_hours=2, priority="critical",
                            note="В первые 120 минут при ИМпST и доступности ЧКВ."),
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
        ProtocolRequirement("medication_class", "antiplatelet",
                            "Двойная антитромбоцитарная терапия (ДАТТ)",
                            priority="critical", med_class="antiplatelet",
                            min_occurrences=2,
                            note="АСК + ингибитор P2Y12 (тикагрелор / клопидогрел / прасугрел)."),
        ProtocolRequirement("medication_class", "anticoag",
                            "Антикоагулянт (НФГ / НМГ / фондапаринукс)",
                            priority="critical", med_class="anticoag"),
        ProtocolRequirement("medication_class", "statin",
                            "Высокоинтенсивный статин",
                            priority="high", med_class="statin"),
        ProtocolRequirement("medication_class", "beta_blocker", "бета-блокатор",
                            priority="high", med_class="beta_blocker"),
        ProtocolRequirement("medication_class", "acei", "ИАПФ / БРА",
                            priority="medium", med_class="acei"),
        ProtocolRequirement("vital", "spo2", "Контроль SpO2", priority="high",
                            note="Оксигенотерапия при SpO2 <90%."),
    ],
)


NSTEMI_PROTOCOL = Protocol(
    code="acs_nstemi",
    name="ОКСбпST / ИМбпST - острый коронарный синдром без подъёма сегмента ST",
    description=(
        "GRACE-ориентированная инвазивная тактика: серия тропонина по протоколу 0/1 ч или 0/3 ч, "
        "ЭКГ в динамике, ЭхоКГ, коронарография в сроки по риску (<=2 ч очень высокий; "
        "<=24 ч высокий). ДАТТ + антикоагулянт, статин, бета-блокатор."
    ),
    requirements=[
        ProtocolRequirement("lab", "troponin_i", "Серия тропонина (0 / 1-3 ч)",
                            window_hours=0, repeat_interval_hours=3, min_occurrences=2,
                            priority="critical"),
        ProtocolRequirement("study", "ecg_12", "Серия ЭКГ 12 отведений", window_hours=0,
                            repeat_interval_hours=6, min_occurrences=2, priority="critical"),
        ProtocolRequirement("study", "echo_cg", "ЭхоКГ", window_hours=24, priority="high"),
        ProtocolRequirement("study", "coronary_angiography", "Коронарография (КГ)",
                            window_hours=24, priority="high",
                            note="В пределах 24 ч при высоком риске ИМбпST."),
        ProtocolRequirement("lab", "creatinine_blood", "Креатинин", window_hours=6,
                            priority="high"),
        ProtocolRequirement("lab", "k_blood", "Калий", window_hours=6, priority="high"),
        ProtocolRequirement("lab", "hgb", "Гемоглобин", window_hours=6, priority="medium"),
        ProtocolRequirement("lab", "ldl", "ЛПНП", window_hours=72, priority="medium"),
        ProtocolRequirement("medication_class", "antiplatelet",
                            "Двойная антитромбоцитарная терапия (ДАТТ)",
                            priority="critical", med_class="antiplatelet", min_occurrences=2),
        ProtocolRequirement("medication_class", "anticoag", "Антикоагулянт",
                            priority="critical", med_class="anticoag"),
        ProtocolRequirement("medication_class", "statin", "Статин",
                            priority="high", med_class="statin"),
        ProtocolRequirement("medication_class", "beta_blocker", "бета-блокатор",
                            priority="high", med_class="beta_blocker"),
        ProtocolRequirement("vital", "spo2", "Контроль SpO2", priority="high"),
    ],
)


UA_PROTOCOL = Protocol(
    code="acs_ua",
    name="НС - нестабильная стенокардия",
    description=(
        "Тропонин-отрицательный ОКС: динамическое наблюдение, серия тропонинов, серия ЭКГ, "
        "ЭхоКГ, антитромбоцитарная терапия (минимум АСК), статин. "
        "Коронарография - при подтверждении ишемии или прогрессировании клиники."
    ),
    requirements=[
        ProtocolRequirement("lab", "troponin_i", "Серия тропонина",
                            window_hours=0, repeat_interval_hours=3, min_occurrences=2,
                            priority="high"),
        ProtocolRequirement("study", "ecg_12", "Серия ЭКГ 12 отведений",
                            window_hours=0, repeat_interval_hours=6, min_occurrences=2,
                            priority="high"),
        ProtocolRequirement("study", "echo_cg", "ЭхоКГ", window_hours=48, priority="medium"),
        ProtocolRequirement("lab", "creatinine_blood", "Креатинин", window_hours=24,
                            priority="medium"),
        ProtocolRequirement("lab", "ldl", "ЛПНП", window_hours=72, priority="medium"),
        ProtocolRequirement("medication_class", "antiplatelet",
                            "Антитромбоцитарная терапия",
                            priority="high", med_class="antiplatelet"),
        ProtocolRequirement("medication_class", "statin", "Статин",
                            priority="medium", med_class="statin"),
        ProtocolRequirement("vital", "spo2", "Контроль SpO2", priority="medium"),
    ],
)


GENERIC_ACS_PROTOCOL = Protocol(
    code="acs_generic",
    name="Подозрение на ОКС - дифференциальная диагностика",
    description=(
        "Минимальный объём обследования при подозрении на ОКС до уточнения "
        "формы (ИМпST / ИМбпST / НС)."
    ),
    requirements=[
        ProtocolRequirement("lab", "troponin_i", "Тропонин",
                            window_hours=0, repeat_interval_hours=3, min_occurrences=2,
                            priority="high"),
        ProtocolRequirement("study", "ecg_12", "ЭКГ 12 отведений", window_hours=0,
                            priority="high", repeat_interval_hours=6, min_occurrences=2),
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


# ---- Маппинг диагностической метки на протокол ----
DIAGNOSIS_TO_PROTOCOL: Dict[AcsDiagnosis, Protocol] = {
    AcsDiagnosis.IM_PST: STEMI_PROTOCOL,
    AcsDiagnosis.OKS_PST: STEMI_PROTOCOL,
    AcsDiagnosis.IM_BPST: NSTEMI_PROTOCOL,
    AcsDiagnosis.OKS_BPST: NSTEMI_PROTOCOL,
    AcsDiagnosis.NS: UA_PROTOCOL,
    AcsDiagnosis.OKS_UNLIKELY: GENERIC_ACS_PROTOCOL,
}


def _icd_group(icd10: str) -> str:
    definition = DIAGNOSIS_BY_ICD.get(icd10.upper())
    return definition.group if definition else ""


def select_protocol(
    diagnoses_icd: List[str] | None,
    ecg_description: str = "",
    triage_category: str = "",
    acs_diagnosis: str | AcsDiagnosis | None = None,
) -> Protocol:
    """Выбрать клинический протокол.

    Приоритет:

    1. Явная диагностическая метка ``acs_diagnosis`` (если передана -
       это результат работы детерминированного классификатора).
    2. МКБ-10 диагнозы, введённые врачом.
    3. Описание ЭКГ (подъём ST / депрессия ST).
    4. ``triage_category``.
    5. Generic ACS-протокол.
    """
    if acs_diagnosis is not None:
        try:
            diag = (
                acs_diagnosis
                if isinstance(acs_diagnosis, AcsDiagnosis)
                else AcsDiagnosis(str(acs_diagnosis))
            )
            return DIAGNOSIS_TO_PROTOCOL.get(diag, GENERIC_ACS_PROTOCOL)
        except ValueError:
            pass

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
    "PROTOCOLS", "DIAGNOSIS_TO_PROTOCOL",
    "select_protocol", "protocol_summary",
]
