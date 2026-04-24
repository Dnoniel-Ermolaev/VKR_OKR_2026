"""Справочники клинических сущностей для пациента с ОКС.

Все каталоги статичны и хранятся в коде. Формат позволяет легко
отдавать их на фронт через JSON-эндпоинт и валидировать Excel-импорт.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional


AcsPriority = Literal["critical", "high", "medium", "low"]
LabGroup = Literal[
    "express", "hematology", "biochem", "coag", "gas", "lipids", "infection", "culture"
]
StudyGroup = Literal["ecg", "imaging", "endoscopy", "other"]
ProcedureGroup = Literal["vascular", "respiratory", "catheter", "rehab", "other"]
MedGroup = Literal[
    "antiplatelet", "anticoag", "beta_blocker", "statin", "acei", "arb",
    "nitrate", "diuretic", "ppi", "analgesic", "antibiotic", "infusion",
    "antiarrhythmic", "other",
]
VitalKind = Literal["pressure", "rate", "temperature", "saturation", "fluid", "other"]
Route = Literal["po", "iv", "iv_drip", "sc", "im", "inhale", "sublingual", "rectal", "topical"]


@dataclass(frozen=True)
class LabDef:
    code: str
    name_ru: str
    unit: str
    group: LabGroup
    acs_priority: AcsPriority = "medium"
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    aliases: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StudyDef:
    code: str
    name_ru: str
    group: StudyGroup
    acs_priority: AcsPriority = "medium"
    aliases: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProcedureDef:
    code: str
    name_ru: str
    group: ProcedureGroup
    acs_priority: AcsPriority = "medium"
    aliases: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MedDef:
    code: str
    name_ru: str
    inn: str
    group: MedGroup
    typical_dose: str = ""
    typical_unit: str = ""
    default_route: Route = "po"
    aliases: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class VitalDef:
    code: str
    name_ru: str
    unit: str
    kind: VitalKind
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None


@dataclass(frozen=True)
class DiagnosisDef:
    icd10: str
    name_ru: str
    group: str = ""


def _as_dict(item) -> dict:
    data = asdict(item)
    return data


def _index(items: list) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for item in items:
        out[item.code if hasattr(item, "code") else item.icd10] = item
    return out


# ----------------------------- VITALS -----------------------------
VITALS: List[VitalDef] = [
    VitalDef("sbp", "АД систолическое", "мм рт.ст.", "pressure", 100, 140, 80, 200),
    VitalDef("dbp", "АД диастолическое", "мм рт.ст.", "pressure", 60, 90, 50, 120),
    VitalDef("hr", "ЧСС", "уд/мин", "rate", 50, 100, 40, 140),
    VitalDef("rr", "ЧД", "в мин", "rate", 12, 20, 8, 30),
    VitalDef("temp", "Температура", "°C", "temperature", 36.0, 37.2, 35.0, 39.0),
    VitalDef("spo2", "SpO2", "%", "saturation", 95, 100, 90, 100),
    VitalDef("diuresis_in", "Диурез внутрь", "мл", "fluid"),
    VitalDef("diuresis_out", "Диурез в/в", "мл", "fluid"),
    VitalDef("stool", "Стул", "раз", "other"),
    VitalDef("glucose_bedside", "Глюкоза бедсайд", "ммоль/л", "other", 3.9, 6.1, 2.5, 15.0),
]


# ----------------------------- LABS -----------------------------
LABS: List[LabDef] = [
    # Экспресс-маркеры повреждения миокарда
    LabDef("troponin_i", "Тропонин I", "нг/мл", "express", "critical",
           ref_low=0.0, ref_high=0.029, critical_high=0.1, aliases=["тропонин", "troponin"]),
    LabDef("troponin_t", "Тропонин T", "нг/мл", "express", "critical",
           ref_low=0.0, ref_high=0.014, critical_high=0.05, aliases=["тропонин т"]),
    LabDef("myoglobin", "Миоглобин", "нг/мл", "express", "high",
           ref_low=0.0, ref_high=90.0),
    LabDef("ck_mb", "КФК-МВ", "Ед/л", "express", "high",
           ref_low=0.0, ref_high=25.0, aliases=["кфк-мв", "ck-mb", "mb"]),
    LabDef("nt_probnp", "NT-proBNP", "пг/мл", "express", "high",
           ref_low=0.0, ref_high=125.0, aliases=["probnp", "bnp"]),
    LabDef("d_dimer", "D-димер", "мкг/мл", "express", "high",
           ref_low=0.0, ref_high=0.5, aliases=["д-димер"]),

    # ОАК
    LabDef("hgb", "Гемоглобин", "г/л", "hematology", "medium",
           ref_low=120, ref_high=160, critical_low=70),
    LabDef("rbc", "Эритроциты", "10^12/л", "hematology", "low",
           ref_low=3.8, ref_high=5.5),
    LabDef("wbc", "Лейкоциты", "10^9/л", "hematology", "medium",
           ref_low=4.0, ref_high=9.0, critical_high=25.0),
    LabDef("plt", "Тромбоциты", "10^9/л", "hematology", "high",
           ref_low=150, ref_high=400, critical_low=50),
    LabDef("hct", "Гематокрит", "%", "hematology", "low",
           ref_low=37, ref_high=52),
    LabDef("esr", "СОЭ", "мм/ч", "hematology", "low",
           ref_low=0, ref_high=20),

    # Биохимия
    LabDef("glucose_blood", "Глюкоза крови", "ммоль/л", "biochem", "high",
           ref_low=3.9, ref_high=6.1, critical_low=2.5, critical_high=15.0,
           aliases=["глюкоза"]),
    LabDef("creatinine_blood", "Креатинин", "мкмоль/л", "biochem", "high",
           ref_low=60, ref_high=110, critical_high=200, aliases=["креатинин"]),
    LabDef("urea_blood", "Мочевина", "ммоль/л", "biochem", "medium",
           ref_low=2.5, ref_high=8.3, aliases=["мочевина"]),
    LabDef("bilirubin_total", "Билирубин общий", "мкмоль/л", "biochem", "low",
           ref_low=3.4, ref_high=20.5, aliases=["билирубин"]),
    LabDef("ast_blood", "АСТ (АсАТ)", "Ед/л", "biochem", "high",
           ref_low=0, ref_high=40, aliases=["асат", "ast"]),
    LabDef("alt_blood", "АЛТ (АлАТ)", "Ед/л", "biochem", "medium",
           ref_low=0, ref_high=40, aliases=["алат", "alt"]),
    LabDef("amylase_blood", "Амилаза крови", "Ед/л", "biochem", "low",
           ref_low=28, ref_high=100),
    LabDef("ldh", "ЛДГ", "Ед/л", "biochem", "medium",
           ref_low=0, ref_high=480, aliases=["лдг"]),
    LabDef("total_protein", "Общий белок", "г/л", "biochem", "low",
           ref_low=64, ref_high=83),

    # Коагулограмма
    LabDef("aptt", "АЧТВ", "сек", "coag", "high",
           ref_low=25, ref_high=35),
    LabDef("inr", "МНО", "", "coag", "high",
           ref_low=0.8, ref_high=1.2, critical_high=4.0, aliases=["мно"]),
    LabDef("pti", "ПТИ", "%", "coag", "medium",
           ref_low=70, ref_high=130, aliases=["пти"]),
    LabDef("fibrinogen", "Фибриноген", "г/л", "coag", "medium",
           ref_low=2.0, ref_high=4.0),

    # КЩС и электролиты
    LabDef("k_blood", "Калий", "ммоль/л", "gas", "critical",
           ref_low=3.5, ref_high=5.1, critical_low=3.0, critical_high=6.0,
           aliases=["калий", "k"]),
    LabDef("na_blood", "Натрий", "ммоль/л", "gas", "high",
           ref_low=135, ref_high=145, critical_low=125, critical_high=155),
    LabDef("cl_blood", "Хлор", "ммоль/л", "gas", "low",
           ref_low=98, ref_high=107),
    LabDef("mg_blood", "Магний", "ммоль/л", "gas", "medium",
           ref_low=0.7, ref_high=1.05),
    LabDef("ca_blood", "Кальций", "ммоль/л", "gas", "medium",
           ref_low=2.15, ref_high=2.55),
    LabDef("ph_blood", "pH крови", "", "gas", "critical",
           ref_low=7.35, ref_high=7.45, critical_low=7.2, critical_high=7.55),
    LabDef("pco2", "pCO2", "мм рт.ст.", "gas", "high",
           ref_low=35, ref_high=45),
    LabDef("po2", "pO2", "мм рт.ст.", "gas", "high",
           ref_low=75, ref_high=100, critical_low=55),
    LabDef("hco3", "HCO3", "ммоль/л", "gas", "medium",
           ref_low=22, ref_high=26),
    LabDef("lactate", "Лактат", "ммоль/л", "gas", "critical",
           ref_low=0.5, ref_high=2.2, critical_high=4.0),

    # Липидный спектр
    LabDef("cholesterol_total", "Холестерин общий", "ммоль/л", "lipids", "medium",
           ref_low=3.0, ref_high=5.2),
    LabDef("ldl", "ЛПНП (LDL)", "ммоль/л", "lipids", "high",
           ref_low=0.0, ref_high=3.4),
    LabDef("hdl", "ЛПВП (HDL)", "ммоль/л", "lipids", "medium",
           ref_low=0.9, ref_high=1.96),
    LabDef("vldl", "ЛПОНП (VLDL)", "ммоль/л", "lipids", "low",
           ref_low=0.2, ref_high=0.9),
    LabDef("triglycerides", "Триглицериды", "ммоль/л", "lipids", "medium",
           ref_low=0.4, ref_high=2.3),
    LabDef("atherogenic_index", "Коэфф. атерогенности", "", "lipids", "low",
           ref_low=1.8, ref_high=3.3),

    # Инфекции
    LabDef("hbsag", "HbsAg (гепатит B)", "", "infection", "low", aliases=["гепатит b"]),
    LabDef("anti_hcv", "Anti-HCV (гепатит C)", "", "infection", "low", aliases=["гепатит c"]),
    LabDef("anti_hiv", "Anti-HIV", "", "infection", "low", aliases=["вич", "hiv"]),
    LabDef("rw", "RW (сифилис)", "", "infection", "low", aliases=["сифилис"]),

    # Посевы
    LabDef("culture_blood", "Посев крови", "", "culture", "low"),
    LabDef("culture_urine", "Посев мочи", "", "culture", "low"),
    LabDef("culture_sputum", "Посев мокроты", "", "culture", "low"),
]


# ----------------------------- STUDIES -----------------------------
STUDIES: List[StudyDef] = [
    StudyDef("ecg_12", "ЭКГ 12-канальная", "ecg", "critical",
             aliases=["экг", "ecg"]),
    StudyDef("ecg_holter", "ЭКГ-МТ (холтер)", "ecg", "medium",
             aliases=["холтер", "суточный экг"]),
    StudyDef("echo_cg", "ЭхоКГ (ЭхоКС)", "imaging", "high",
             aliases=["эхокг", "эхокс", "эхо-кг"]),
    StudyDef("coronary_angiography", "Коронарография (СКГ)", "imaging", "critical",
             aliases=["скг", "коронарография"]),
    StudyDef("chest_xray", "R-графия органов грудной клетки", "imaging", "medium",
             aliases=["r-графия", "рентген"]),
    StudyDef("abdominal_us", "УЗИ органов брюшной полости", "imaging", "low",
             aliases=["узи брюшной"]),
    StudyDef("mri_cardiac", "МРТ сердца", "imaging", "medium"),
    StudyDef("ct_chest", "МСКТ грудной клетки", "imaging", "medium",
             aliases=["мскт"]),
    StudyDef("ct_coronary", "МСКТ-коронарография", "imaging", "high"),
    StudyDef("spirometry", "Спирометрия (ФВД)", "other", "low", aliases=["фвд"]),
]


# ----------------------------- PROCEDURES -----------------------------
PROCEDURES: List[ProcedureDef] = [
    ProcedureDef("pci_stent", "ЧКВ со стентированием", "vascular", "critical",
                 aliases=["стентирование", "чкв", "pci"]),
    ProcedureDef("pci_balloon", "Баллонная ангиопластика", "vascular", "critical"),
    ProcedureDef("thromboaspiration", "Тромбоаспирация", "vascular", "high"),
    ProcedureDef("thrombolysis", "Тромболизис", "vascular", "critical",
                 aliases=["тромболитик"]),
    ProcedureDef("iabp", "ВАБК (внутриаортальная контрпульсация)", "vascular", "high",
                 aliases=["вабк"]),
    ProcedureDef("cabg", "АКШ", "vascular", "high", aliases=["акш"]),
    ProcedureDef("mechanical_recanalization", "Механическая реканализация", "vascular", "high",
                 aliases=["мр"]),
    ProcedureDef("oxygen_therapy", "Оксигенотерапия (O2)", "respiratory", "high",
                 aliases=["o2-терапия", "кислород"]),
    ProcedureDef("mechanical_ventilation", "ИВЛ", "respiratory", "critical",
                 aliases=["ивл"]),
    ProcedureDef("nip_ventilation", "НИВЛ", "respiratory", "medium"),
    ProcedureDef("peripheral_catheter", "Постановка ПВК", "catheter", "low",
                 aliases=["пвк"]),
    ProcedureDef("central_catheter", "Постановка ЦВК", "catheter", "medium",
                 aliases=["цвк"]),
    ProcedureDef("urinary_catheter", "Уретральный катетер", "catheter", "low"),
    ProcedureDef("defibrillation", "Дефибрилляция", "vascular", "critical"),
    ProcedureDef("cardioversion", "Электрическая кардиоверсия", "vascular", "high"),
    ProcedureDef("pacing_temporary", "Временная ЭКС", "vascular", "high"),
    ProcedureDef("pleural_puncture", "Плевральная пункция", "other", "medium"),
    ProcedureDef("physiotherapy", "ЛФК", "rehab", "low", aliases=["лфк"]),
    ProcedureDef("massage", "Массаж", "rehab", "low"),
    ProcedureDef("compression_stockings", "Компрессионный трикотаж", "rehab", "low"),
]


# ----------------------------- MEDICATIONS -----------------------------
MEDICATIONS: List[MedDef] = [
    # Антиагреганты
    MedDef("asa", "Ацекардол (аспирин)", "Ацетилсалициловая кислота", "antiplatelet",
           "100", "мг", "po", aliases=["aspirin", "ацекардол", "тромбо асс"]),
    MedDef("clopidogrel", "Плавикс (клопидогрел)", "Клопидогрел", "antiplatelet",
           "75", "мг", "po", aliases=["plavix", "клопидогрел"]),
    MedDef("ticagrelor", "Brilinta (тикагрелор)", "Тикагрелор", "antiplatelet",
           "90", "мг", "po", aliases=["brilinta", "тикагрелор", "брилинта"]),
    MedDef("prasugrel", "Эффиент (прасугрел)", "Прасугрел", "antiplatelet",
           "10", "мг", "po"),

    # Антикоагулянты
    MedDef("heparin_uf", "Гепарин (НФГ)", "Гепарин натрия", "anticoag",
           "5000", "ЕД", "iv", aliases=["гепарин", "heparin"]),
    MedDef("enoxaparin", "Клексан (эноксапарин)", "Эноксапарин", "anticoag",
           "1", "мг/кг", "sc", aliases=["клексан"]),
    MedDef("fondaparinux", "Арикстра (фондапаринукс)", "Фондапаринукс", "anticoag",
           "2.5", "мг", "sc"),
    MedDef("warfarin", "Варфарин", "Варфарин", "anticoag",
           "5", "мг", "po"),
    MedDef("rivaroxaban", "Ксарелто (ривароксабан)", "Ривароксабан", "anticoag",
           "20", "мг", "po"),

    # Бета-блокаторы
    MedDef("metoprolol", "Метопролол", "Метопролол", "beta_blocker",
           "50", "мг", "po"),
    MedDef("bisoprolol", "Бисопролол", "Бисопролол", "beta_blocker",
           "5", "мг", "po"),
    MedDef("carvedilol", "Карведилол", "Карведилол", "beta_blocker",
           "12.5", "мг", "po"),

    # Статины
    MedDef("atorvastatin", "Аторвастатин", "Аторвастатин", "statin",
           "80", "мг", "po"),
    MedDef("rosuvastatin", "Розувастатин", "Розувастатин", "statin",
           "20", "мг", "po"),

    # иАПФ / БРА
    MedDef("enalapril", "Эналаприл", "Эналаприл", "acei",
           "5", "мг", "po"),
    MedDef("perindopril", "Периндоприл", "Периндоприл", "acei",
           "5", "мг", "po"),
    MedDef("losartan", "Лозартан", "Лозартан", "arb",
           "50", "мг", "po"),
    MedDef("valsartan", "Валсартан", "Валсартан", "arb",
           "80", "мг", "po"),

    # Нитраты
    MedDef("nitroglycerin", "Нитроглицерин", "Нитроглицерин", "nitrate",
           "0.5", "мг", "sublingual"),
    MedDef("isosorbide_dn", "Динизорб (изосорбид динитрат)", "Изосорбида динитрат",
           "nitrate", "2", "мг/час", "iv_drip", aliases=["динизорб"]),

    # Диуретики
    MedDef("furosemide", "Фуросемид", "Фуросемид", "diuretic",
           "40", "мг", "iv"),
    MedDef("torasemide", "Diuver (торасемид)", "Торасемид", "diuretic",
           "10", "мг", "po", aliases=["diuver", "диувер"]),
    MedDef("spironolactone", "Верошпирон (спиронолактон)", "Спиронолактон",
           "diuretic", "25", "мг", "po", aliases=["верошпирон"]),

    # ИПП
    MedDef("omeprazole", "Омепразол", "Омепразол", "ppi",
           "20", "мг", "po"),
    MedDef("pantoprazole", "Пантопразол", "Пантопразол", "ppi",
           "40", "мг", "iv"),

    # Анальгетики
    MedDef("morphine", "Морфин", "Морфин", "analgesic",
           "5", "мг", "iv"),
    MedDef("fentanyl", "Фентанил", "Фентанил", "analgesic",
           "0.05", "мг", "iv"),
    MedDef("ketorolac", "Кеторолак", "Кеторолак", "analgesic",
           "30", "мг", "iv"),

    # Антибиотики
    MedDef("ceftriaxone", "Цефтриаксон", "Цефтриаксон", "antibiotic",
           "1.0", "г", "iv"),
    MedDef("amoxiclav", "Амоксиклав", "Амоксициллин+клавуланат", "antibiotic",
           "1.2", "г", "iv"),

    # Инфузии
    MedDef("saline", "NaCl 0.9%", "Натрия хлорид", "infusion",
           "400", "мл", "iv_drip"),
    MedDef("ringer", "Раствор Рингера", "Рингер", "infusion",
           "400", "мл", "iv_drip"),
    MedDef("glucose5", "Глюкоза 5%", "Глюкоза", "infusion",
           "400", "мл", "iv_drip"),

    # Антиаритмики
    MedDef("amiodarone", "Амиодарон (кордарон)", "Амиодарон", "antiarrhythmic",
           "150", "мг", "iv", aliases=["кордарон"]),
    MedDef("lidocaine", "Лидокаин", "Лидокаин", "antiarrhythmic",
           "80", "мг", "iv"),
]


# ----------------------------- DIAGNOSES (ICD-10) -----------------------------
DIAGNOSES: List[DiagnosisDef] = [
    # Острый коронарный синдром / инфаркт
    DiagnosisDef("I21.0", "Острый трансмуральный инфаркт миокарда передней стенки", "STEMI"),
    DiagnosisDef("I21.1", "Острый трансмуральный инфаркт миокарда нижней стенки", "STEMI"),
    DiagnosisDef("I21.2", "Острый трансмуральный инфаркт миокарда других локализаций", "STEMI"),
    DiagnosisDef("I21.4", "Острый субэндокардиальный инфаркт миокарда", "NSTEMI"),
    DiagnosisDef("I21.9", "Острый инфаркт миокарда неуточнённый", "ACS"),
    DiagnosisDef("I20.0", "Нестабильная стенокардия", "UA"),
    DiagnosisDef("I20.8", "Другие формы стенокардии", "stable_angina"),
    DiagnosisDef("I22.0", "Повторный инфаркт миокарда передней стенки", "STEMI"),
    DiagnosisDef("I22.1", "Повторный инфаркт миокарда нижней стенки", "STEMI"),
    DiagnosisDef("I24.9", "Острая ишемическая болезнь сердца неуточнённая", "ACS"),
    DiagnosisDef("I25.0", "Атеросклеротическая болезнь сердца", "chronic_cad"),
    DiagnosisDef("I25.1", "Атеросклеротическая болезнь сердца", "chronic_cad"),
    DiagnosisDef("I25.2", "Перенесённый в прошлом инфаркт миокарда", "post_mi"),

    # Осложнения и коморбидности
    DiagnosisDef("I50.0", "Застойная сердечная недостаточность", "heart_failure"),
    DiagnosisDef("I50.1", "Левожелудочковая недостаточность", "heart_failure"),
    DiagnosisDef("I50.9", "ХСН неуточнённая", "heart_failure"),
    DiagnosisDef("I10", "Эссенциальная (первичная) гипертензия", "hypertension"),
    DiagnosisDef("I11.9", "Гипертензивная болезнь сердца", "hypertension"),
    DiagnosisDef("I48.0", "Пароксизмальная фибрилляция предсердий", "arrhythmia"),
    DiagnosisDef("I48.1", "Постоянная фибрилляция предсердий", "arrhythmia"),
    DiagnosisDef("I49.9", "Нарушение сердечного ритма неуточнённое", "arrhythmia"),
    DiagnosisDef("I46.9", "Остановка сердца неуточнённая", "cardiac_arrest"),
    DiagnosisDef("I71.0", "Расслоение аорты", "aorta"),

    # Сопутствующие
    DiagnosisDef("E11.9", "Сахарный диабет 2 типа", "diabetes"),
    DiagnosisDef("E78.0", "Чистая гиперхолестеринемия", "dyslipidemia"),
    DiagnosisDef("N18.9", "ХБП неуточнённая", "kidney"),
    DiagnosisDef("J44.9", "ХОБЛ неуточнённая", "copd"),
]


# ----------------------------- INDEXES -----------------------------
VITAL_BY_CODE: Dict[str, VitalDef] = {item.code: item for item in VITALS}
LAB_BY_CODE: Dict[str, LabDef] = {item.code: item for item in LABS}
STUDY_BY_CODE: Dict[str, StudyDef] = {item.code: item for item in STUDIES}
PROCEDURE_BY_CODE: Dict[str, ProcedureDef] = {item.code: item for item in PROCEDURES}
MEDICATION_BY_CODE: Dict[str, MedDef] = {item.code: item for item in MEDICATIONS}
DIAGNOSIS_BY_ICD: Dict[str, DiagnosisDef] = {item.icd10: item for item in DIAGNOSES}


def _build_alias_index(items, code_attr: str = "code") -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for item in items:
        code = getattr(item, code_attr)
        idx[code.lower()] = code
        idx[getattr(item, "name_ru", "").strip().lower()] = code
        for alias in getattr(item, "aliases", []) or []:
            idx[str(alias).strip().lower()] = code
    idx.pop("", None)
    return idx


LAB_ALIAS_INDEX = _build_alias_index(LABS)
STUDY_ALIAS_INDEX = _build_alias_index(STUDIES)
PROCEDURE_ALIAS_INDEX = _build_alias_index(PROCEDURES)
MEDICATION_ALIAS_INDEX = _build_alias_index(MEDICATIONS)
VITAL_ALIAS_INDEX: Dict[str, str] = {item.code.lower(): item.code for item in VITALS}
VITAL_ALIAS_INDEX.update({item.name_ru.lower(): item.code for item in VITALS})


def resolve_lab(token: str) -> Optional[str]:
    if not token:
        return None
    return LAB_ALIAS_INDEX.get(str(token).strip().lower())


def resolve_study(token: str) -> Optional[str]:
    if not token:
        return None
    return STUDY_ALIAS_INDEX.get(str(token).strip().lower())


def resolve_procedure(token: str) -> Optional[str]:
    if not token:
        return None
    return PROCEDURE_ALIAS_INDEX.get(str(token).strip().lower())


def resolve_medication(token: str) -> Optional[str]:
    if not token:
        return None
    return MEDICATION_ALIAS_INDEX.get(str(token).strip().lower())


def resolve_vital(token: str) -> Optional[str]:
    if not token:
        return None
    return VITAL_ALIAS_INDEX.get(str(token).strip().lower())


def flag_for_lab(code: str, value: Optional[float]) -> str:
    """Return 'norm' | 'low' | 'high' | 'critical_low' | 'critical_high' | 'unknown'."""
    if value is None:
        return "unknown"
    definition = LAB_BY_CODE.get(code)
    if definition is None:
        return "unknown"
    if definition.critical_low is not None and value <= definition.critical_low:
        return "critical_low"
    if definition.critical_high is not None and value >= definition.critical_high:
        return "critical_high"
    if definition.ref_low is not None and value < definition.ref_low:
        return "low"
    if definition.ref_high is not None and value > definition.ref_high:
        return "high"
    return "norm"


def flag_for_vital(code: str, value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    definition = VITAL_BY_CODE.get(code)
    if definition is None:
        return "unknown"
    if definition.critical_low is not None and value <= definition.critical_low:
        return "critical_low"
    if definition.critical_high is not None and value >= definition.critical_high:
        return "critical_high"
    if definition.ref_low is not None and value < definition.ref_low:
        return "low"
    if definition.ref_high is not None and value > definition.ref_high:
        return "high"
    return "norm"


def catalog_as_json() -> dict:
    return {
        "vitals": [_as_dict(v) for v in VITALS],
        "labs": [_as_dict(v) for v in LABS],
        "studies": [_as_dict(v) for v in STUDIES],
        "procedures": [_as_dict(v) for v in PROCEDURES],
        "medications": [_as_dict(v) for v in MEDICATIONS],
        "diagnoses": [_as_dict(v) for v in DIAGNOSES],
    }


__all__ = [
    "LabDef", "StudyDef", "ProcedureDef", "MedDef", "VitalDef", "DiagnosisDef",
    "VITALS", "LABS", "STUDIES", "PROCEDURES", "MEDICATIONS", "DIAGNOSES",
    "VITAL_BY_CODE", "LAB_BY_CODE", "STUDY_BY_CODE", "PROCEDURE_BY_CODE",
    "MEDICATION_BY_CODE", "DIAGNOSIS_BY_ICD",
    "resolve_lab", "resolve_study", "resolve_procedure", "resolve_medication", "resolve_vital",
    "flag_for_lab", "flag_for_vital", "catalog_as_json",
]
