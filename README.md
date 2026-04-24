# LangGraph ACS: reproducible demo

Локальный прототип системы поддержки решений при ОКС с `web UI`, case-based режимом, Excel-импортом, protocol-driven контролем пациента и локальным `RAG` по клиническим рекомендациям.

Этот `README` ориентирован не на обзор архитектуры, а на воспроизводимый сценарий:

1. скачать репозиторий;
2. поднять зависимости;
3. инициализировать PostgreSQL;
4. построить или проверить индекс `RAG`;
5. открыть `web UI`;
6. прогнать готовые demo-кейсы и увидеть систему в действии.

## Что можно проверить после запуска

После установки преподаватель или проверяющий сможет явно проверить:

- быструю оценку ОКС на вкладке `Оценка ОКС`;
- case-based режим на вкладке `Пациент`;
- Excel-импорт структурированных clinical data;
- автоматическую `reassess` после импорта;
- protocol-driven dashboard (`STEMI / NSTEMI / UA`);
- историю оценок и генерацию клинического отчета;
- локальный `RAG` с `citations`.

## Что входит в репозиторий

- `data/guidelines/` — тексты рекомендаций для `RAG`;
- `data/chroma/` — локальное хранилище индекса `Chroma` (если уже собрано);
- `examples/demo_cases/` — готовые `.xlsx`-сценарии для демонстрации;
- `src/web/` — `FastAPI` + `web UI`;
- `src/core/` — triage-граф, report-граф и workflow runner;
- `src/medical/` — clinical rules, catalog, protocols;
- `src/infrastructure/` — `db`, importers, `rag`.

## Быстрый старт

### 1. Клонирование и виртуальное окружение

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd VKR_OKR_2026
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

### 2. Настройка переменных окружения

Скопируйте `.env.example` в `.env` и при необходимости поправьте значения под свою локальную среду.

Пример:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/acs_db
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_MODEL_B=qwen2.5:3b-instruct
```

## PostgreSQL

Проект использует PostgreSQL для хранения:

- пациентов;
- визитов;
- clinical cases;
- observations, studies, procedures, medications, diagnoses;
- history assessments;
- clinical reports.

### 1. Создайте пустую БД

Название по умолчанию в примере: `acs_db`.

### 2. Инициализируйте схему

Первичный запуск:

```bash
python -m src.infrastructure.db.init_db
```

Этот скрипт:

- создаёт таблицы;
- добавляет тестовых пациентов, если база пуста.

Если база уже старая и после обновления проекта не хватает колонок, выполните:

```bash
python -m scripts.upgrade_db_schema
```

Скрипт `scripts.migrate_v2` нужен только для миграции старых tracking-данных из предыдущей версии проекта. Для чистого запуска он не обязателен.

### 3. Альтернатива: восстановление готовой demo-базы

Если в репозитории или рядом с ним приложен готовый дамп, например
`examples/demo_db/demo_acs_db.dump`, PostgreSQL можно не наполнять вручную, а
сразу восстановить текущее demo-состояние проекта.

#### Вариант A. Через `pgAdmin 4`

1. Откройте `pgAdmin 4`.
2. Подключитесь к своему PostgreSQL server.
3. Создайте пустую базу `acs_db`, если её ещё нет:
   - `Databases` -> `Create` -> `Database...`
4. Нажмите правой кнопкой по базе `acs_db`.
5. Выберите `Restore...`.
6. Укажите файл дампа, например:
   `examples/demo_db/demo_acs_db.dump`
7. Если формат не определился автоматически, выберите `Custom`.
8. Нажмите `Restore`.

После этого в базе появятся:

- demo-пациенты;
- визиты;
- кейсы;
- observations;
- studies;
- procedures;
- medications;
- diagnoses;
- assessments;
- reports.

#### Вариант B. Через `pg_restore`

Если у вас установлен PostgreSQL client tools, можно восстановить дамп из командной строки.

Сначала создайте пустую базу:

```bash
createdb -U postgres -h localhost -p 5432 acs_db
```

Затем выполните восстановление:

```bash
pg_restore -U postgres -h localhost -p 5432 -d acs_db --clean --if-exists examples/demo_db/demo_acs_db.dump
```

> На Windows `pg_restore.exe` может лежать, например, в
> `C:\Program Files\PostgreSQL\18\bin\pg_restore.exe` или
> `C:\Program Files\PostgreSQL\18\pgAdmin 4\runtime\pg_restore.exe`.
> Если команда `pg_restore` не находится по имени, укажите полный путь к `.exe`.

## Ollama и LLM

Для полной демонстрации желательно установить и запустить `Ollama`.

Рекомендуемые модели:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b-instruct
```

Если `Ollama` недоступна, часть логики всё равно будет работать в fallback-режиме, но:

- free-text парсинг станет слабее;
- роутеры будут использовать fallback-ветки;
- `RAG + LLM` интерпретация и отчет будут менее показательны.

## RAG: что нужно сделать

`RAG` в проекте опирается на:

- `data/guidelines/*.txt` — корпус рекомендаций;
- `data/chroma/` — локальный индекс `Chroma PersistentClient`.

### Нормальный вариант запуска

Просто выполните:

```bash
python -m src.infrastructure.rag.rag_setup
```

Этот скрипт:

- убеждается, что в `data/guidelines/` есть хотя бы seed-документ;
- очищает guideline-файлы в удобный для chunking вид;
- строит индекс `Chroma` в `data/chroma/`.

### Как понять, что RAG готов

После запуска `rag_setup` в консоли должны появиться сообщения вида:

- `Guideline seed ready: ...`
- `Indexed chunks: N`

Если `Indexed chunks > 0`, retrieval готов к работе.

## Запуск web UI

```bash
python -m uvicorn src.web.api:app --reload
```

После запуска откройте:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

## Явная проверка проекта в деле

Ниже минимальный воспроизводимый сценарий, который показывает не только интерфейс, но и реальную работу workflow.

### Проверка 1. Быстрый triage на вкладке `Оценка ОКС`

Откройте `examples/demo_cases/presentation_scenarios.json` и возьмите один из сценариев.

Проще всего начать с `structured_high_risk_assess`.

Что сделать:

1. Открыть вкладку `Оценка ОКС`.
2. Ввести поля пациента вручную:
   - имя;
   - тип боли;
   - описание ЭКГ;
   - тропонин;
   - ЧСС;
   - АД.
3. Нажать `Получить результат`.

Что ожидаемо увидеть:

- `risk_level = high`
- `triage_category = high_risk_fast_track`
- текстовое `explanation`
- при неочевидном случае — `citations` из `RAG`

### Проверка 2. Case-based режим через Excel

Основные demo-файлы лежат в `examples/demo_cases/`:

- `structured_high_risk_demo_fixed.xlsx`
- `protocol_gap_demo.xlsx`
- `patient_card_stemi_case.xlsx`

#### Сценарий A. Быстрый high-risk кейс

Файл:

`examples/demo_cases/structured_high_risk_demo_fixed.xlsx`

Что сделать:

1. Выбрать пациента.
2. Создать визит.
3. Нажать `+ Новый кейс`.
4. Перейти на вкладку `Excel`.
5. Импортировать `structured_high_risk_demo_fixed.xlsx`.

Что ожидаемо увидеть:

- автоматическую `reassess`;
- `risk_level = high`;
- `triage_category = high_risk_fast_track`;
- протокол `STEMI`;
- частично заполненный protocol dashboard;
- `completion_percent` около `70.6`.

#### Сценарий B. Демонстрация protocol gaps

Файл:

`examples/demo_cases/protocol_gap_demo.xlsx`

Что сделать:

1. Создать новый чистый кейс.
2. Импортировать `protocol_gap_demo.xlsx`.
3. Открыть вкладку `Пациент`.

Что ожидаемо увидеть:

- высокий риск уже распознан;
- протокол выбран как `STEMI`;
- completion низкий, около `5.9`;
- видны `critical_pending`;
- в `alerts` есть пропущенные критические шаги и отсутствие части терапии.

Этот сценарий показывает, что система делает не только triage, но и clinical control полноты ведения пациента.

#### Сценарий C. Почти полный кейс по реальной карте

Файл:

`examples/demo_cases/patient_card_stemi_case.xlsx`

Что сделать:

1. Создать новый кейс.
2. Импортировать `patient_card_stemi_case.xlsx`.
3. Открыть вкладки `Анализы`, `Исследования`, `Процедуры`, `Назначения`, `Диагнозы`, `Оценки`, `Отчеты`.

Что ожидаемо увидеть:

- содержательный structured-case;
- протокол `STEMI`;
- высокий риск;
- `completion_percent = 100.0`;
- пустой `critical_pending`;
- возможность сгенерировать клинический отчет.

Подробности по demo-материалам описаны в:

`examples/demo_cases/README.md`

## Что именно демонстрирует проект

### 1. Triage workflow

Система умеет:

- принимать structured input;
- принимать free-text и парсить его через `LLM`;
- валидировать полноту данных;
- запускать rule-based clinical assessment;
- при необходимости подключать `RAG + LLM`;
- сохранять risk, explanation, citations и route metadata.

### 2. Case lifecycle

Поддерживаются:

- создание кейса;
- переоценка (`reassess`);
- закрытие и переоткрытие;
- накопление structured-data внутри кейса.

### 3. Protocol-driven control

Для кейса строится clinical dashboard по протоколам:

- `STEMI`
- `NSTEMI`
- `UA`

Система показывает:

- выбранный протокол;
- completion;
- pending critical steps;
- alerts.

### 4. Clinical report generation

Для кейса можно запустить отдельный report workflow и получить клинический текст с опорой на состояние кейса и `RAG`-контекст.

## Как интерпретировать наличие RAG в проекте

`RAG` в проекте используется не как главный механизм всей системы, а как дополнительный knowledge-layer.

Практически это означает:

- очевидные high-risk случаи могут быть обработаны правилами без необходимости в `RAG`;
- неочевидные и пограничные случаи получают retrieval по рекомендациям;
- report workflow также использует retrieval для генерации более обоснованного текста.

## Команды для локальной проверки без UI

### CLI: structured input

```bash
python -m src.cli.main \
  --mode single \
  --model qwen2.5:7b-instruct \
  --name Ivan \
  --age 67 \
  --gender male \
  --pain-type typical \
  --ecg-changes "ST-elevation V2-V5" \
  --troponin 0.17 \
  --hr 78 \
  --bp 135/92 \
  --symptoms-text "давящая боль за грудиной" \
  --output data/structured_check.json
```

### CLI: free-text input

```bash
python -m src.cli.main \
  --mode single \
  --model qwen2.5:7b-instruct \
  --free-text "Мужчина 67 лет, жгучая боль за грудиной, ST-elevation V2-V5, тропонин 22.712, ЧСС 78, АД 161/89, SpO2 94, креатинин 145" \
  --output data/free_text_check.json \
  --force-llm
```

> На Windows PowerShell перенос строк делается через `` ` ``, а не через `\`.

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
│   └── upgrade_db_schema.py
├── src/
│   ├── cli/
│   ├── core/
│   ├── evaluation/
│   ├── infrastructure/
│   ├── medical/
│   └── web/
├── tests/
│   └── unit/
├── tmp_runs/
├── README.md
└── requirements.txt
```

## Краткая архитектура

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
```

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

## На что обратить внимание при проверке

- `reassess` работает и без предварительной быстрой оценки, если кейс постепенно наполняется structured-данными;
- `protocol-control` и triage — это разные уровни логики;
- `reopen` не равен `reassess`: переоткрытие меняет статус кейса, а не заново оценивает пациента;
- `RAG` может не срабатывать в очевидных high-risk cases, потому что workflow может завершить решение на rule-based ветке;
- для максимально чистой демонстрации лучше каждый `.xlsx` импортировать в новый пустой кейс.

## Если что-то не заработало

### Приложение стартует, но таблиц нет

Запустить:

```bash
python -m src.infrastructure.db.init_db
python -m scripts.upgrade_db_schema
```

### RAG не отдаёт citations

Проверить:

1. что в `data/guidelines/` есть `.txt`-документы;
2. что после `python -m src.infrastructure.rag.rag_setup` индексировались чанки;
3. что установлен `chromadb` и `sentence-transformers`;
4. что проверяется не только очевидный `rule-only` кейс, а и более неопределённый сценарий.

### LLM-ветки не работают

Проверить:

1. установлен ли `ollama`;
2. запущен ли сервис `Ollama`;
3. загружены ли модели `qwen2.5:7b-instruct` и `qwen2.5:3b-instruct`.

## Что ещё можно сделать потом

- расширить связность между triage-payload и structured-case сущностями;
- углубить protocol-driven контроль;
- усилить explainability;
- провести более формальную clinical validation;
- улучшить `RAG` и evaluation pipeline.

## Ограничения

- это исследовательский прототип, а не медицинское изделие;
- не использовать для реальной диагностики и назначения лечения;
- любые clinical conclusions требуют проверки врачом.
