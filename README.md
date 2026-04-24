# LangGraph ACS (Web + Case Prototype)

Локальный прототип системы triage и сопровождения пациента с ОКС для учебных и исследовательских задач. Проект поддерживает быстрый `triage`-запуск, web-интерфейс для case-based работы, локальный RAG по клиническим рекомендациям, структурированный ввод данных пациента и генерацию клинического отчета.

## Что сделано сейчас

- `LangGraph`-граф для оценки риска ОКС с многошаговым роутингом.
- Поддержка двух режимов входа:
  - `structured`-поля пациента;
  - `free-text` с LLM/эвристическим парсингом.
- Три LLM-router узла с `confidence-gates` и fallback-маршрутизацией.
- Локальный RAG на `Chroma` + `sentence-transformers` embeddings + cross-encoder reranking.
- Case lifecycle в БД:
  - `start_case`
  - `resume_case`
  - `reassess`
  - `close`
  - `reopen`
- Структурированные клинические сущности:
  - витальные показатели
  - лабораторные анализы
  - исследования
  - процедуры
  - назначения
  - диагнозы
- Protocol-driven контроль пациента по сценариям `STEMI / NSTEMI / UA`.
- Отдельный `report_graph` для генерации эпикриза/клинического отчета.
- Web UI с вкладками пациента, активного кейса, динамики наблюдений и Excel-импорта.
- Excel-импорт и генерация шаблонов для `Vitals / Labs / Studies / Procedures / Medications / Diagnoses`.
- Набор unit/smoke тестов для каталогов, протоколов, lifecycle, Excel и API.
- Модуль evaluation для расчета `sensitivity`, `specificity`, `AUC`.

## Архитектура продукта

```mermaid
flowchart TD
    A[Web UI]
    B[FastAPI]
    C[Case API]
    D[Excel Import]
    E[SQL Repository]
    F[PostgreSQL]
    G[Triage Graph]
    H[Case Control]
    I[Report Graph]
    J[Local RAG]

    A --> B
    B --> C
    B --> D

    C --> E
    D --> E
    E --> F

    C --> G
    C --> H
    C --> I

    G --> J
    I --> J

    G --> E
    I --> E
    E --> H
```

### Что происходит в пользовательском сценарии

1. Пользователь открывает web UI и выбирает пациента/визит.
2. Система создает или переиспользует активный кейс.
3. Данные поступают через формы, CRUD-операции или Excel-импорт.
4. `workflow_runner` собирает `patient_data`, запускает triage-граф и сохраняет оценку.
5. `patient_control` выбирает протокол ОКС и считает прогресс, overdue-пункты и alerts.
6. При необходимости запускается отдельный `report_graph` для эпикриза.

## Архитектура triage-графа

```mermaid
flowchart TD
    A[Input: structured or free text] --> B{Free text}

    B --> C[llm_parse_history]
    B --> D[parse_input]

    C --> D
    D --> E[router_pretriage]

    E --> F[clarify_data]
    E --> G[rule_check]

    F --> C
    F --> H[data_quality_issue]

    G --> I[router_diagnostic]

    I --> J[high_risk_fast_track]
    I --> K[diagnostic_uncertain]
    I --> L[low_risk_observation]

    K --> M[rag_retrieval]
    M --> N[llm_assess]

    J --> O[router_management]
    L --> O
    N --> O

    O --> P[monitor_plan]
    O --> Q[recommend_treatment]
    O --> R[output_save]

    P --> R
    Q --> R
    H --> R
```

## Логика ветвления

- `router_pretriage`: достаточно ли данных для безопасного продолжения.
- `router_diagnostic`: вести ли кейс в `urgent`, `rag_llm` или `rule_only`.
- `router_management`: нужен ли мониторинг, лечебные подсказки или финализация.
- Если `confidence` роутера низкий, система переключается в безопасную fallback-ветку.
- Критичные поля (`troponin`, `ecg_changes`, `hr`, `bp`) контролируются отдельно.

## RAG в текущей реализации

- База рекомендаций хранится локально в `data/guidelines/*.txt`.
- Retriever использует `Chroma PersistentClient`.
- Dense retrieval строится на `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Поверх семантического поиска включен lexical scoring.
- Для топ-кандидатов используется cross-encoder reranking:
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- В ответах сохраняются `citations` и информация о странице/чанке.

## Web UI и API

Web-интерфейс находится в `src/web/` и поддерживает:

- быструю вкладку `Оценка ОКС`;
- карточку пациента и историю визитов;
- активный кейс с риском, протоколом и completion;
- CRUD для `Vitals / Labs / Studies / Procedures / Medications / Diagnoses`;
- Excel template download и Excel import;
- историю оценок и клинические отчеты.

Ключевые API-эндпоинты:

- `POST /api/assess`
- `POST /api/cases/start`
- `POST /api/cases/{id}/resume`
- `POST /api/cases/{id}/reassess`
- `POST /api/cases/{id}/close`
- `POST /api/cases/{id}/reopen`
- `POST /api/cases/{id}/report`
- `GET /api/cases/{id}/control`
- `GET /api/catalog`
- `POST /api/cases/{id}/excel-import`

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.infrastructure.rag.rag_setup
```

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b-instruct
```

## Настройка базы данных

Нужно:

1. Установить PostgreSQL.
2. Создать пустую базу `acs_db`.
3. Указать `DATABASE_URL` в `.env`.

Пример:

```env
DATABASE_URL=postgresql://postgres:ВАШ_ПАРОЛЬ@localhost:5432/acs_db
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_MODEL_B=qwen2.5:3b-instruct
```

Первичная инициализация:

```bash
python -m src.infrastructure.db.init_db
```

## Запуск web-интерфейса

```bash
python -m uvicorn src.web.api:app --reload
```

После запуска откройте [http://127.0.0.1:8000](http://127.0.0.1:8000).

## CLI-запуск

### Structured

```bash
python -m src.cli.main \
  --mode single \
  --model qwen2.5:7b-instruct \
  --name Ivan \
  --age 67 \
  --gender male \
  --pain-type typical \
  --ecg-changes "ST-depression" \
  --troponin 0.18 \
  --hr 108 \
  --bp 145/90 \
  --spo2 95 \
  --creatinine 145 \
  --glucose 7.8 \
  --killip-class II \
  --echo-dkg-results "ФВ 60%" \
  --symptoms-text "жгучая боль за грудиной" \
  --output data/structured_extended.json \
  --force-llm
```

### Free-text

```bash
python -m src.cli.main \
  --mode single \
  --model qwen2.5:7b-instruct \
  --free-text "Мужчина 67 лет, жгучая боль за грудиной, ST-elevation V2-V5, тропонин 22.712, ЧСС 78, АД 161/89, SpO2 94, креатинин 145" \
  --output data/free_text_result.json \
  --force-llm
```

### A/B

```bash
python -m src.cli.main \
  --mode ab \
  --model qwen2.5:7b-instruct \
  --model-b qwen2.5:3b-instruct \
  --name Ivan \
  --pain-type typical \
  --ecg-changes "ST-depression" \
  --troponin 0.12 \
  --hr 102 \
  --bp 130/85 \
  --symptoms-text "давящая боль в груди 30 минут" \
  --output data/ab_result.json \
  --force-llm
```

PowerShell использует перенос строки через `` ` ``, а не `\`.

## Что возвращает система

- `risk`, `risk_level`, `explanation`
- `triage_category`, `next_step`
- `route_confidence`, `route_reason`
- `parse_confidence`, `missing_fields`
- `llm_used`
- `record_id`

## Демонстрационные материалы

Готовые презентационные артефакты находятся в `examples/demo_cases/`:

- реальные `.xlsx`-кейсы для импорта через UI;
- JSON/markdown-описания тестовых сценариев;
- кейс, близкий к реальной карте пациента из `Итог.png`;
- ожидаемые outcome для демонстрации в презентации.

## Структура проекта

```text
.
├── data/
│   ├── chroma/
│   ├── guidelines/
│   ├── ocr/
│   └── patients.csv
├── examples/
│   └── demo_cases/
├── scripts/
│   ├── migrate_v2.py
│   ├── run.sh
│   └── test.sh
├── src/
│   ├── cli/
│   ├── core/                  # graph, report_graph, workflow runner, prompts, tools
│   ├── evaluation/            # метрики и evaluation runner
│   ├── infrastructure/        # db, importers, rag
│   ├── medical/               # catalog, protocols, rules, scores
│   └── web/                   # API, services, static, templates
├── tests/
│   └── unit/
├── tmp_runs/
├── README.md
└── requirements.txt
```

## Что пока не сделано

- Раздел аналитики в web UI.
- Полноценный датасет с формально оформленными результатами evaluation.
- Внешняя клиническая валидация на реальном размеченном наборе случаев.
- Автоматизированный сбор клинических данных из внешних hospital systems.

## Ограничения и дисклеймер

- Это исследовательский прототип, не медицинское изделие.
- Не использовать для реальной диагностики и назначения лечения.
- Любой вывод требует клинической верификации врачом.
