// Тип Object - изменяется при выборе пациента.
// Хранит базовую информацию, требуемую для отображения на форме
let currentPatient = null;
let currentCaseId = null;
let currentCaseDetails = null;
let currentCaseControl = null;
let currentReportPreview = null;
let currentCaseSubTab = 'vitals';
let medicalCatalog = null;
let excelImportReport = null;
let caseBusy = false;
let caseBusyMessage = '';

const CASE_SUBTABS = [
    { id: 'vitals',       label: 'Витальные' },
    { id: 'labs',         label: 'Анализы' },
    { id: 'studies',      label: 'Исследования' },
    { id: 'procedures',   label: 'Процедуры' },
    { id: 'medications',  label: 'Назначения' },
    { id: 'diagnoses',    label: 'Диагнозы' },
    { id: 'assessments',  label: 'Оценки' },
    { id: 'reports',      label: 'Отчёты' },
    { id: 'excel',        label: 'Excel' },
];

// Заполнение списка пациентов при загрузке приложения
async function loadPatientsFromDB() {
    const listContainer = document.getElementById('patientList');
    listContainer.innerHTML = "<p style='text-align:center; color:#94a3b8;'>Загрузка...</p>";

    try {
        // Делаем GET запрос к нашему Python API
        const response = await fetch('/api/patients');
        const patients = await response.json();

        // Очищаем контейнер
        listContainer.innerHTML = "";

        // Генерируем HTML для каждого пациента из базы
// Генерируем HTML для каждого пациента из базы
        patients.forEach(p => {
            const cardHTML = `
                <div class="patient-card" id="card-${p.id}" style="border-left-color: ${p.risk_color};" onclick="selectPatient(${p.id})">
                    <div class="patient-card-header">
                        <span style="font-weight: bold; color: #3b82f6;">${p.display_id}</span>
                        <span>${p.birth_date} г.р.</span>
                    </div>
                    <div class="patient-card-name">${p.full_name}</div>
                </div>
            `;
            listContainer.insertAdjacentHTML('beforeend', cardHTML);
        });
    } catch (error) {
        listContainer.innerHTML = "<p style='color:red;'>Ошибка подключения к БД</p>";
    }
}

// Логика переключения вкладок
function switchTab(tabId, btnElement) {

    // Прячем все вкладки
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    // Показываем нужную
    document.getElementById(tabId).classList.add('active');
    btnElement.classList.add('active');
}

// Логика поиска пациентов
function searchPatient() {

    // Получаем то, что ввел пользователь, и убираем лишние пробелы по краям
    const rawInput = document.getElementById('searchInput').value.trim();
    const searchMessage = document.getElementById('searchMessage');
    
    // Получаем список всех карточек пациентов на странице
    const cards = document.querySelectorAll('.patient-card');

    // Если поле пустое - показываем всех пациентов обратно и прячем сообщение
    if (rawInput === "") {
        cards.forEach(card => card.style.display = "block");
        searchMessage.style.display = "none";
        return;
    }

    const query = rawInput.toLowerCase(); // Переводим ввод в нижний регистр
    
    // Определяем тип поиска: по ID (цифры) или по ФИО (буквы) 
    // charAt(0) берет первый символ. Регулярка /^\d$/ проверяет, цифра ли это.
    const isIdSearch = /^\d$/.test(query.charAt(0)); 
    
    let matchCount = 0; // Счетчик найденных пациентов

    // Проходимся циклом по каждой карточке
    cards.forEach(card => {

        const idElement = card.querySelector('.patient-card-header span:first-child');
        const nameElement = card.querySelector('.patient-card-name');
        
        let matchFound = false;

        if (isIdSearch) {
            // Ищем по ID. Извлекаем только цифры из текста "ID: 10042" -> "10042"
            const rawIdText = idElement.textContent;        // "ID: 10042"
            const idNumber = rawIdText.replace(/\D/g, '');  // Удаляем всё, кроме цифр -> "10042"
            
            if (idNumber.startsWith(query)) {
                matchFound = true;
            }

        } else {
            // Ищем по ФИО (просто проверяем, есть ли подстрока)
            const nameText = nameElement.textContent.toLowerCase();
            if (nameText.includes(query)) {
                matchFound = true;
            }
        }

        // Показываем или прячем карточку
        if (matchFound) {
            card.style.display = "block";
            matchCount++;
        } else {
            card.style.display = "none";
        }
    });

    // Обработка результатов поиска
    if (matchCount === 0) {
        const searchType = isIdSearch ? "ID" : "ФИО";
        searchMessage.textContent = `Пациент с ${searchType} "${rawInput}" не найден.`;
        searchMessage.style.display = "block";
    } else {
        searchMessage.style.display = "none";
    }
}

// Поиск по нажатию клавиши Enter
document.getElementById('searchInput').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Отменяем стандартное поведение формы
        searchPatient();        // Запускаем нашу функцию
    }
});

// Отправка веб-формы в Python
function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function renderAssessmentCitations(citations) {
    const panel = document.getElementById('resultCitationsPanel');
    const list = document.getElementById('resultCitationsList');
    if (!panel || !list) return;

    if (!Array.isArray(citations) || citations.length === 0) {
        panel.style.display = 'none';
        list.innerHTML = '<div class="citation-empty">Для этого результата источники RAG не использовались.</div>';
        return;
    }

    panel.style.display = 'block';
    list.innerHTML = citations
        .map(item => `<div class="citation-item">${escapeHtml(item)}</div>`)
        .join('');
}

function getAssessmentFormPayload() {
    const name = document.getElementById('ptName')?.value.trim() || currentPatient?.full_name || '';
    const painType = document.getElementById('ptPain')?.value || 'typical';
    const ecgChanges = document.getElementById('ptEcg')?.value.trim() || 'unknown';
    const troponinRaw = parseFloat(document.getElementById('ptTrop')?.value);
    const hrRaw = parseInt(document.getElementById('ptHr')?.value, 10);
    const bp = document.getElementById('ptBp')?.value.trim() || '120/80';

    return {
        name,
        pain_type: painType,
        ecg_changes: ecgChanges,
        troponin: Number.isFinite(troponinRaw) ? troponinRaw : 0,
        hr: Number.isFinite(hrRaw) ? hrRaw : 70,
        bp,
        free_text: "",
    };
}

function syncAssessmentFormFromCase(casePayload) {
    if (!casePayload || typeof casePayload !== 'object') return;

    const nameInput = document.getElementById('ptName');
    const painInput = document.getElementById('ptPain');
    const ecgInput = document.getElementById('ptEcg');
    const troponinInput = document.getElementById('ptTrop');
    const hrInput = document.getElementById('ptHr');
    const bpInput = document.getElementById('ptBp');

    if (nameInput && casePayload.name) nameInput.value = casePayload.name;
    if (painInput && casePayload.pain_type) painInput.value = casePayload.pain_type;
    if (ecgInput && casePayload.ecg_changes) ecgInput.value = casePayload.ecg_changes;
    if (troponinInput && casePayload.troponin !== undefined && casePayload.troponin !== null) {
        troponinInput.value = casePayload.troponin;
    }
    if (hrInput && casePayload.hr !== undefined && casePayload.hr !== null) {
        hrInput.value = casePayload.hr;
    }
    if (bpInput && casePayload.bp) bpInput.value = casePayload.bp;
}

async function runAssessment() {
    const outputBox = document.getElementById('resultOutput');
    renderAssessmentCitations([]);
    outputBox.innerText = "Загрузка...";

    // Собираем те же данные, которые затем можно сохранить как case.
    const payload = getAssessmentFormPayload();

    // Отправляем JSON на наш Python сервер (FastAPI)
    const response = await fetch('/api/assess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    // Печатаем ответ
    const result = await response.json();
    renderAssessmentCitations(result.citations || []);
    outputBox.innerText = JSON.stringify(result, null, 2);
}

// Отправка команды из Консоли
async function runConsole() {
    const outputBox = document.getElementById('consoleOutput');
    outputBox.innerText = "Выполнение команды...";

    const commandText = document.getElementById('consoleInput').value;

    const response = await fetch('/api/console', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: commandText })
    });

    const result = await response.json();
    outputBox.innerText = JSON.stringify(result, null, 2);
}

// Логика изменения размера панели
document.addEventListener('DOMContentLoaded', async function () {
    const resizer = document.getElementById('dragMe');
    const leftSide = document.getElementById('sidebar');

    if (!resizer || !leftSide) return;

    const mouseDownHandler = function (e) {
        // Добавляем класс всему ТЕЛУ страницы
        // Это мгновенно запрещает выделение любого текста в любой части экрана
        document.body.classList.add('dragging-active');

        document.addEventListener('mousemove', mouseMoveHandler);
        document.addEventListener('mouseup', mouseUpHandler);
        
        resizer.classList.add('active');
    };

    const mouseMoveHandler = function (e) {
        // Вычисляем ширину по положению мыши
        let newWidth = e.clientX;
        
        // Ограничители (чтобы не схлопнуть в 0 и не растянуть на весь экран)
        if (newWidth < 200) newWidth = 200;
        if (newWidth > window.innerWidth * 0.6) newWidth = window.innerWidth * 0.6;
        
        leftSide.style.width = `${newWidth}px`;
    };

    const mouseUpHandler = function () {
        // Убираем класс с тела страницы - выделение снова работает
        document.body.classList.remove('dragging-active');
        
        resizer.classList.remove('active');
        document.removeEventListener('mousemove', mouseMoveHandler);
        document.removeEventListener('mouseup', mouseUpHandler);
    };

    resizer.addEventListener('mousedown', mouseDownHandler);
    
    // Подгрузка пациентов
    await loadPatientsFromDB();
});


// Логика выбора пациента
async function selectPatient(patientId) {
    document.getElementById('visitsPanel').innerHTML = "Загрузка данных пациента...";
    
    try {
        const response = await fetch(`/api/patients/${patientId}`);
        const patient = await response.json();

        if (patient.error) throw new Error(patient.error);

        currentPatient = patient;
        const caseIds = (patient.cases || []).map(item => item.id);
        if (!currentCaseId || !caseIds.includes(currentCaseId)) {
            currentCaseId = caseIds.length ? caseIds[0] : null;
        }
        currentReportPreview = null;
        excelImportReport = null;
        if (currentCaseId) {
            await refreshActiveCase();
        } else {
            currentCaseDetails = null;
            currentCaseControl = null;
        }
        const nameInput = document.getElementById('ptName');
        if (nameInput) {
            nameInput.value = patient.full_name;
        }

        updateSelectionUI(patientId);
        renderPatientDashboard();
        renderHospitalDashboard();

    } catch (error) {
        console.error("Ошибка при получении деталей:", error);
        alert("Не удалось загрузить данные пациента");
    }
}

// Визуальное обновление
function updateSelectionUI(patientId) {
    document.querySelectorAll('.patient-card').forEach(card => card.classList.remove('selected'));
    const activeCard = document.getElementById(`card-${patientId}`);
    if (activeCard) activeCard.classList.add('selected');

    document.getElementById('activePatientText').innerText = 
        `Выбран пациент: [ID: ${currentPatient.id}] ${currentPatient.full_name}`;
    document.getElementById('unselectBtn').style.display = 'block';
}

 // Вывести приложение из режима работы с конкретным человеком и вернуть его в нейтральное состояние
function unselectPatient() {
    currentPatient = null;
    currentCaseId = null;
    currentCaseDetails = null;
    currentCaseControl = null;
    currentReportPreview = null;
    excelImportReport = null;

    document.querySelectorAll('.patient-card').forEach(card => card.classList.remove('selected'));
    document.getElementById('activePatientText').innerText = 'Пациент не выбран';
    document.getElementById('unselectBtn').style.display = 'none';
    renderPatientDashboard();
    renderHospitalDashboard();
}

function renderPatientDashboard() {
    const visitsPanel = document.getElementById('visitsPanel');
    const statsPanel = document.getElementById('statsPanel');

    if (!currentPatient) {
        visitsPanel.innerHTML = '<div class="empty-state">Выберите пациента для просмотра визитов</div>';
        statsPanel.innerHTML = '<div class="empty-state">Выберите пациента для работы с визитами</div>';
        return;
    }

    visitsPanel.innerHTML = renderVisitsPanel();
    statsPanel.innerHTML = renderVisitContextPanel();
}

function renderVisitsPanel() {
    const visits = currentPatient.visits || [];
    const visitsHtml = visits.length
        ? [...visits].reverse().map(v => `
            <div class="list-row">
                <span><b>📅 ${escapeHtml(v.date)}</b></span>
                <button onclick="deleteVisit(${v.id})" class="icon-btn" title="Удалить визит">🗑️</button>
            </div>
        `).join('')
        : "<p class='empty-inline'>Визитов пока нет</p>";

    return `
        <div class="section-title-row">
            <h4 style="margin: 0; color: #334155;">История визитов</h4>
            <button class="small-btn" onclick="openVisitModal()">+ Визит</button>
        </div>
        <div class="info-card">
            <div><b>${escapeHtml(currentPatient.full_name)}</b></div>
            <div class="muted-line">ID: ${currentPatient.display_id || currentPatient.id} | Дата рождения: ${escapeHtml(currentPatient.birth_date)}</div>
        </div>
        ${visitsHtml}
    `;
}

function renderVisitContextPanel() {
    const visits = currentPatient.visits || [];
    const latestVisit = visits.length ? visits[visits.length - 1] : null;

    return `
        <div class="section-title-row">
            <h4 style="margin: 0; color: #334155;">Карточка визитов</h4>
        </div>
        <div class="info-card">
            <div><b>${escapeHtml(currentPatient.full_name)}</b></div>
            <div class="muted-line">Всего визитов: ${visits.length}</div>
            <div class="muted-line">Последний визит: ${latestVisit ? escapeHtml(latestVisit.date) : 'нет данных'}</div>
        </div>
        <div class="empty-inline">
            Раздел оставлен для будущей логики единичных визитов. Постоянное наблюдение, анализы,
            исследования, переоценка и эпикриз перенесены во вкладку «Стационар».
        </div>
    `;
}

function renderHospitalDashboard() {
    const casesListPanel = document.getElementById('casesListPanel');
    const caseDetailsPanel = document.getElementById('caseDetailsPanel');
    if (!casesListPanel || !caseDetailsPanel) return;

    if (!currentPatient) {
        casesListPanel.innerHTML = '<div class="empty-state">Выберите пациента для просмотра кейсов</div>';
        caseDetailsPanel.innerHTML = '<div class="empty-state">Выберите кейс из списка слева</div>';
        return;
    }

    casesListPanel.innerHTML = renderCasesListPanel();
    caseDetailsPanel.innerHTML = renderCasePanel();
}

function renderCasesListPanel() {
    const cases = [...(currentPatient.cases || [])].sort((a, b) => {
        const statusOrder = { active: 0, awaiting_labs: 1, completed: 2 };
        const byStatus = (statusOrder[a.status] ?? 3) - (statusOrder[b.status] ?? 3);
        if (byStatus !== 0) return byStatus;
        return new Date(b.created_at || 0) - new Date(a.created_at || 0);
    });

    const casesHtml = cases.length
        ? cases.map(item => {
            const activeClass = currentCaseId === item.id ? 'case-card active' : 'case-card';
            const statusTagClass = item.status === 'completed'
                ? 'case-status-closed'
                : item.status === 'awaiting_labs'
                    ? 'case-status-waiting'
                    : 'case-status-open';
            return `
                <div class="${activeClass}" onclick="selectCase('${item.id}')">
                    <div class="case-card-title">${escapeHtml(item.title || item.id)}</div>
                    <div class="case-card-meta">
                        <span class="case-status-tag ${statusTagClass}">${escapeHtml(caseStatusLabel(item.status))}</span>
                        <span>${escapeHtml(item.latest_risk_level || '')}</span>
                    </div>
                    <div class="muted-line">Создан: ${escapeHtml(formatDt(item.created_at))}</div>
                    <div class="muted-line">Этап: ${escapeHtml(item.current_stage || '—')}</div>
                </div>
            `;
        }).join('')
        : "";

    return `
        <div class="section-title-row">
            <h4 style="margin: 0; color: #334155;">Кейсы</h4>
            <button class="small-btn" onclick="createNewCase()">+ Новый</button>
        </div>
        ${casesHtml}
    `;
}

function caseStatusLabel(status) {
    if (status === 'active') return 'Открыт';
    if (status === 'awaiting_labs') return 'Ожидает анализы';
    if (status === 'completed') return 'Закрыт';
    return status || '—';
}

function caseStatusClass(status) {
    if (status === 'active') return 'active';
    if (status === 'awaiting_labs') return 'warn';
    if (status === 'completed') return 'done';
    return 'neutral';
}

function renderCasePanel() {
    if (!currentCaseDetails || !currentCaseDetails.case) {
        return `
            <div class="empty-state">
                Выберите существующий кейс слева или создайте новый кнопкой «+ Новый кейс».
            </div>
        `;
    }
    const info = currentCaseDetails.case || {};
    const summary = currentCaseControl?.summary || {};
    const protocol = currentCaseDetails.protocol || currentCaseControl?.protocol || null;
    const completion = Math.round(summary.completion_percent ?? 0);
    const alerts = Array.isArray(summary.alerts) ? summary.alerts : [];
    const alertsHtml = alerts.length
        ? alerts.map(a => `<div class="alert-item">${escapeHtml(a)}</div>`).join('')
        : '<div class="empty-inline">Alert-сигналов нет</div>';

    const closed = info.status === 'completed';
    const disabledAttr = caseBusy ? 'disabled' : '';
    const lifecycleButtons = closed
        ? `<button class="small-btn" onclick="reopenCase()" ${disabledAttr}>Переоткрыть</button>`
        : `<button class="small-btn" onclick="reassessCase()" ${disabledAttr}>Переоценить</button>
           <button class="small-btn" onclick="generateCaseReport()" ${disabledAttr}>Эпикриз</button>
           <button class="small-btn" onclick="closeCase()" ${disabledAttr}>Закрыть</button>`;
    const busyHtml = caseBusy
        ? `<div class="case-busy-panel">
               <span class="case-spinner"></span>
               <span>${escapeHtml(caseBusyMessage || 'Команда получена, идут вычисления...')}</span>
           </div>`
        : '';

    return `
        <div class="case-banner case-banner-${caseStatusClass(info.status)}">
            <div class="case-banner-top">
                <div>
                    <div class="case-banner-title">${escapeHtml(info.title || info.id)}</div>
                    <div class="muted-line">ID ${escapeHtml(info.id)} | Статус: <b>${escapeHtml(info.status || '—')}</b> | Этап: ${escapeHtml(info.current_stage || '—')}</div>
                </div>
                <div class="case-action-row">
                    <div class="button-row">
                        ${lifecycleButtons}
                    </div>
                    <button class="small-btn danger-soft" onclick="deleteCase()" ${disabledAttr}>Удалить</button>
                </div>
            </div>
            ${busyHtml}
            <div class="case-banner-bottom">
                <div class="metric-card"><span class="metric-label">Риск</span><span class="metric-value">${escapeHtml(info.latest_risk_level || '—')}</span></div>
                <div class="metric-card"><span class="metric-label">Категория</span><span class="metric-value">${escapeHtml(info.latest_triage_category || '—')}</span></div>
                <div class="metric-card"><span class="metric-label">Протокол</span><span class="metric-value">${escapeHtml(protocol?.name || '—')}</span></div>
                <div class="metric-card"><span class="metric-label">Готовность</span>
                    <span class="metric-value">${completion}%</span>
                </div>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="width:${completion}%"></div></div>
            <div class="alerts-box">${alertsHtml}</div>
        </div>

        <div class="subtab-bar">
            ${CASE_SUBTABS.map(tab => `
                <button class="subtab-btn ${currentCaseSubTab === tab.id ? 'active' : ''}"
                        onclick="setCaseSubTab('${tab.id}')">${escapeHtml(tab.label)}</button>
            `).join('')}
        </div>
        <div id="caseSubPanel">${renderCaseSubPanel()}</div>
    `;
}

function renderCaseSubPanel() {
    switch (currentCaseSubTab) {
        case 'vitals':       return renderVitalsSubTab();
        case 'labs':         return renderLabsSubTab();
        case 'studies':      return renderStudiesSubTab();
        case 'procedures':   return renderProceduresSubTab();
        case 'medications':  return renderMedicationsSubTab();
        case 'diagnoses':    return renderDiagnosesSubTab();
        case 'assessments':  return renderAssessmentsSubTab();
        case 'reports':      return renderReportsSubTab();
        case 'excel':        return renderExcelSubTab();
        default:             return '';
    }
}

function setCaseSubTab(id) {
    currentCaseSubTab = id;
    renderHospitalDashboard();
}

function setCaseBusy(isBusy, message = '') {
    caseBusy = isBusy;
    caseBusyMessage = message;
    renderHospitalDashboard();
}

async function runCaseBusyAction(message, action) {
    if (caseBusy) return null;
    setCaseBusy(true, message);
    try {
        return await action();
    } finally {
        setCaseBusy(false, '');
    }
}

function renderVitalsSubTab() {
    const items = (currentCaseDetails?.observations || []).filter(o => o.category === 'vital');
    const rowsHtml = items.length
        ? items.map(o => `
            <tr>
                <td>${escapeHtml(formatDt(o.recorded_at))}</td>
                <td><b>${escapeHtml(o.name)}</b><div class="muted-line">${escapeHtml(o.code)}</div></td>
                <td class="flag-${o.flag || 'unknown'}">${escapeHtml(fmtNum(o.value_num))} ${escapeHtml(o.unit || '')}</td>
                <td>${escapeHtml(rangeText(o.ref_low, o.ref_high))}</td>
                <td><button class="icon-btn" onclick="deleteObservation(${o.id}, 'vital')" title="Удалить">🗑️</button></td>
            </tr>
        `).join('')
        : `<tr><td colspan="5" class="empty-inline">Витальные показатели пока не введены</td></tr>`;
    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Витальные показатели (hourly)</h4>
                <button class="small-btn" onclick="openObservationModal('vital')">+ Добавить</button>
            </div>
            <table class="case-table">
                <thead><tr><th>Время</th><th>Показатель</th><th>Значение</th><th>Норма</th><th></th></tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
    `;
}

function renderLabsSubTab() {
    const items = (currentCaseDetails?.observations || []).filter(o => o.category === 'lab');
    const rowsHtml = items.length
        ? items.map(o => `
            <tr>
                <td>${escapeHtml(formatDt(o.recorded_at))}</td>
                <td><b>${escapeHtml(o.name)}</b><div class="muted-line">${escapeHtml(o.code)}</div></td>
                <td class="flag-${o.flag || 'unknown'}">${escapeHtml(fmtNum(o.value_num) || o.value_text || '—')} ${escapeHtml(o.unit || '')}</td>
                <td>${escapeHtml(rangeText(o.ref_low, o.ref_high))}</td>
                <td>${escapeHtml(flagLabel(o.flag))}</td>
                <td><button class="icon-btn" onclick="deleteObservation(${o.id}, 'lab')" title="Удалить">🗑️</button></td>
            </tr>
        `).join('')
        : `<tr><td colspan="6" class="empty-inline">Анализы пока не внесены</td></tr>`;
    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Лабораторные анализы</h4>
                <button class="small-btn" onclick="openObservationModal('lab')">+ Добавить анализ</button>
            </div>
            <table class="case-table">
                <thead><tr><th>Время</th><th>Анализ</th><th>Значение</th><th>Норма</th><th>Флаг</th><th></th></tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
    `;
}

function renderStudiesSubTab() {
    const items = currentCaseDetails?.studies || [];
    const rowsHtml = items.length
        ? items.map(s => `
            <tr>
                <td>${escapeHtml(formatDt(s.started_at))}</td>
                <td><b>${escapeHtml(s.name)}</b><div class="muted-line">${escapeHtml(s.code)}</div></td>
                <td><span class="pill pill-${statusClass(s.status)}">${escapeHtml(s.status)}</span></td>
                <td>${escapeHtml(s.result_text || '—')}</td>
                <td><button class="icon-btn" onclick="deleteEntity('study', ${s.id})" title="Удалить">🗑️</button></td>
            </tr>
        `).join('')
        : `<tr><td colspan="5" class="empty-inline">Исследований пока нет</td></tr>`;
    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Инструментальные исследования</h4>
                <button class="small-btn" onclick="openEntityModal('study')">+ Добавить</button>
            </div>
            <table class="case-table">
                <thead><tr><th>Начато</th><th>Исследование</th><th>Статус</th><th>Результат</th><th></th></tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
    `;
}

function renderProceduresSubTab() {
    const items = currentCaseDetails?.procedures || [];
    const rowsHtml = items.length
        ? items.map(p => `
            <tr>
                <td>${escapeHtml(formatDt(p.started_at))}</td>
                <td><b>${escapeHtml(p.name)}</b><div class="muted-line">${escapeHtml(p.code)}</div></td>
                <td><span class="pill pill-${statusClass(p.status)}">${escapeHtml(p.status)}</span></td>
                <td>${escapeHtml(p.operator || '—')}</td>
                <td><button class="icon-btn" onclick="deleteEntity('procedure', ${p.id})" title="Удалить">🗑️</button></td>
            </tr>
        `).join('')
        : `<tr><td colspan="5" class="empty-inline">Процедур пока нет</td></tr>`;
    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Процедуры и вмешательства</h4>
                <button class="small-btn" onclick="openEntityModal('procedure')">+ Добавить</button>
            </div>
            <table class="case-table">
                <thead><tr><th>Начато</th><th>Процедура</th><th>Статус</th><th>Оператор</th><th></th></tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
    `;
}

function renderMedicationsSubTab() {
    const items = currentCaseDetails?.medications || [];
    const rowsHtml = items.length
        ? items.map(m => `
            <tr>
                <td><b>${escapeHtml(m.name)}</b><div class="muted-line">${escapeHtml(m.code || '')}</div></td>
                <td>${escapeHtml(m.med_class || '—')}</td>
                <td>${escapeHtml(m.dose || '')} ${escapeHtml(m.unit || '')}</td>
                <td>${escapeHtml(m.route || '')}</td>
                <td>${escapeHtml(m.frequency || '')}</td>
                <td><span class="pill pill-${statusClass(m.status)}">${escapeHtml(m.status)}</span></td>
                <td><button class="icon-btn" onclick="deleteEntity('medication', ${m.id})" title="Удалить">🗑️</button></td>
            </tr>
        `).join('')
        : `<tr><td colspan="7" class="empty-inline">Назначений пока нет</td></tr>`;
    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Медикаментозные назначения</h4>
                <button class="small-btn" onclick="openEntityModal('medication')">+ Добавить</button>
            </div>
            <table class="case-table">
                <thead><tr><th>Препарат</th><th>Класс</th><th>Доза</th><th>Путь</th><th>Кратность</th><th>Статус</th><th></th></tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
    `;
}

function renderDiagnosesSubTab() {
    const items = currentCaseDetails?.diagnoses || [];
    const rowsHtml = items.length
        ? items.map(d => `
            <tr>
                <td><b>${escapeHtml(d.icd10)}</b></td>
                <td>${escapeHtml(d.name)}</td>
                <td>${escapeHtml(d.diagnosis_type || '')}</td>
                <td>${escapeHtml(formatDt(d.established_at))}</td>
                <td><button class="icon-btn" onclick="deleteEntity('diagnosis', ${d.id})" title="Удалить">🗑️</button></td>
            </tr>
        `).join('')
        : `<tr><td colspan="5" class="empty-inline">Диагнозов пока нет</td></tr>`;
    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Диагнозы (МКБ-10)</h4>
                <button class="small-btn" onclick="openEntityModal('diagnosis')">+ Добавить</button>
            </div>
            <table class="case-table">
                <thead><tr><th>МКБ</th><th>Диагноз</th><th>Тип</th><th>Установлен</th><th></th></tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        </div>
    `;
}

function renderAssessmentsSubTab() {
    const assessments = currentCaseDetails?.assessments || [];
    const tracking = currentCaseControl?.tracking || currentCaseDetails?.tracking || [];
    const trackingHtml = tracking.length
        ? tracking.map(t => `
            <div class="tracking-item ${escapeHtml(t.priority || 'medium')} ${t.status === 'done' ? 'done' : (t.overdue ? 'overdue' : '')}">
                <div class="tracking-main">
                    <span><b>${escapeHtml(t.title)}</b></span>
                    <span>${escapeHtml(t.status)} (${t.done_count}/${t.needed_count})</span>
                </div>
                <div class="muted-line">${escapeHtml(t.kind)} · ${escapeHtml(t.code)} ${t.window_hours ? '· окно ' + t.window_hours + 'ч' : ''}</div>
                ${t.note ? `<div class="tracking-result">${escapeHtml(t.note)}</div>` : ''}
            </div>
        `).join('')
        : '<div class="empty-inline">Трекинг протокола пуст</div>';

    const assessHtml = assessments.length
        ? assessments.slice().reverse().map(a => `
            <div class="list-row">
                <span><b>${escapeHtml(a.run_kind)}</b> · ${escapeHtml(a.risk_level || '')} · ${escapeHtml(a.triage_category || '')}</span>
                <span>${escapeHtml(formatDt(a.created_at))}</span>
            </div>
            ${a.explanation ? `<div class="muted-line" style="margin-bottom:8px;">${escapeHtml(a.explanation)}</div>` : ''}
        `).join('')
        : '<div class="empty-inline">Оценок пока не было</div>';

    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Протокол: ${escapeHtml(currentCaseDetails?.protocol?.name || '—')}</h4>
                <button class="small-btn" onclick="reassessCase()">Запустить переоценку</button>
            </div>
            <div>${trackingHtml}</div>
            <div class="section-title-row" style="margin-top:16px;"><h4 style="margin:0;">История оценок LLM/графа</h4></div>
            <div>${assessHtml}</div>
        </div>
    `;
}

function renderReportsSubTab() {
    const reports = currentCaseDetails?.reports || [];
    const reportsHtml = reports.length
        ? reports.map(r => `
            <div class="report-card">
                <div class="section-title-row">
                    <span><b>${escapeHtml(r.report_type)}</b></span>
                    <span>${escapeHtml(formatDt(r.created_at))}</span>
                </div>
                <pre class="console-output mini-output">${escapeHtml(r.content || '')}</pre>
            </div>
        `).join('')
        : '<div class="empty-inline">Отчётов ещё нет</div>';
    const previewHtml = currentReportPreview
        ? `<pre class="console-output mini-output">${escapeHtml(currentReportPreview.content || JSON.stringify(currentReportPreview, null, 2))}</pre>`
        : '';
    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Эпикриз / отчёт</h4>
                <button class="small-btn" onclick="generateCaseReport()">Сгенерировать</button>
            </div>
            ${previewHtml}
            ${reportsHtml}
        </div>
    `;
}

function renderExcelSubTab() {
    const caseId = currentCaseId;
    const reportHtml = excelImportReport
        ? renderExcelReport(excelImportReport)
        : '<div class="empty-inline">Загрузите .xlsx со структурированными листами Vitals/Labs/Studies/Procedures/Medications/Diagnoses</div>';
    return `
        <div class="subtab-panel">
            <div class="section-title-row">
                <h4 style="margin:0;">Импорт Excel</h4>
                <a class="small-btn" href="/api/cases/${caseId}/excel-template" download>Скачать полный шаблон</a>
            </div>
            <div class="excel-templates">
                ${['Vitals','Labs','Studies','Procedures','Medications','Diagnoses'].map(s =>
                    `<a class="small-btn ghost" href="/api/cases/${caseId}/excel-template/${s}" download>${s}</a>`
                ).join(' ')}
            </div>
            <div class="stack-form" style="margin-top:16px;">
                <input type="file" id="excelFile" accept=".xlsx">
                <div class="button-row">
                    <button class="small-btn" onclick="uploadExcel(true)">Dry-run</button>
                    <button class="small-btn" onclick="uploadExcel(false)">Импортировать</button>
                </div>
            </div>
            <div>${reportHtml}</div>
        </div>
    `;
}

function renderExcelReport(report) {
    if (report.error) {
        return `<div class="alert-item">${escapeHtml(report.error)}</div>`;
    }
    const sheetsHtml = Object.entries(report.sheets || {}).map(([name, info]) => {
        const rejected = (info.rejected || []).map(r => `
            <li class="muted-line">строка ${r.row}: ${escapeHtml(r.reason)}</li>
        `).join('');
        return `
            <div class="excel-sheet-report">
                <b>${escapeHtml(name)}:</b> импортировано ${info.imported} из ${info.total_rows}
                ${rejected ? `<ul>${rejected}</ul>` : ''}
            </div>
        `;
    }).join('');
    return `
        <div class="excel-report">
            <div><b>Итого импортировано:</b> ${report.imported_total} ${report.dry_run ? '(dry-run)' : ''}</div>
            ${sheetsHtml}
        </div>
    `;
}

function renderJsonPreview(value) {
    return JSON.stringify(value, null, 2);
}

async function apiJson(url, options = {}) {
    const response = await fetch(url, options);
    return response.json();
}

async function refreshActiveCase() {
    if (!currentCaseId) {
        currentCaseDetails = null;
        currentCaseControl = null;
        return;
    }
    currentCaseDetails = await apiJson(`/api/cases/${currentCaseId}`);
    currentCaseControl = await apiJson(`/api/cases/${currentCaseId}/control`);
    syncAssessmentFormFromCase(currentCaseDetails?.case?.latest_payload);
}

async function selectCase(caseId) {
    currentCaseId = caseId;
    currentReportPreview = null;
    excelImportReport = null;
    await refreshActiveCase();
    renderHospitalDashboard();
}

async function ensureCatalog() {
    if (medicalCatalog) return medicalCatalog;
    medicalCatalog = await apiJson('/api/catalog');
    return medicalCatalog;
}

async function createNewCase() {
    if (!currentPatient) {
        alert('Сначала выберите пациента.');
        return;
    }
    const payload = {
        patient_id: currentPatient.id,
        ...getAssessmentFormPayload(),
        name: currentPatient.full_name,
        symptoms_text: 'Создано из UI (структурированный ввод)',
        // Кнопка называется "Новый кейс", поэтому не переиспользуем старый active case.
        reuse_active: false,
        llm_model: 'qwen2.5:7b-instruct',
    };
    const result = await apiJson('/api/cases/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (result.error) {
        alert(`Ошибка создания кейса: ${result.error}`);
        return;
    }
    if (result.reused_existing) {
        // toast-like info
        console.info('Возобновили активный кейс:', result.case_id);
    }
    currentCaseId = result.case_id;
    await selectPatient(currentPatient.id);
    currentCaseSubTab = 'vitals';
    renderHospitalDashboard();
}

async function reassessCase() {
    if (!currentCaseId) return;
    await runCaseBusyAction('Команда получена, идет переоценка риска...', async () => {
        const result = await apiJson(`/api/cases/${currentCaseId}/reassess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ llm_model: 'qwen2.5:7b-instruct' }),
        });
        if (result.error) {
            alert(`Ошибка переоценки: ${result.error}`);
            return;
        }
        await refreshActiveCase();
    });
    renderHospitalDashboard();
}

async function closeCase() {
    if (!currentCaseId) return;
    if (!confirm('Закрыть кейс? Его можно будет снова открыть.')) return;
    const result = await apiJson(`/api/cases/${currentCaseId}/close`, { method: 'POST' });
    if (result.error) { alert(result.error); return; }
    await selectPatient(currentPatient.id);
}

async function reopenCase() {
    if (!currentCaseId) return;
    const result = await apiJson(`/api/cases/${currentCaseId}/reopen`, { method: 'POST' });
    if (result.error) { alert(result.error); return; }
    await selectPatient(currentPatient.id);
}

async function deleteCase() {
    if (!currentCaseId || !currentPatient) return;
    if (!confirm('Удалить кейс и все связанные с ним данные? Это действие необратимо.')) return;
    const deletedCaseId = currentCaseId;
    const result = await apiJson(`/api/cases/${deletedCaseId}`, { method: 'DELETE' });
    if (result.error) { alert(result.error); return; }
    currentCaseId = null;
    currentCaseDetails = null;
    currentCaseControl = null;
    currentReportPreview = null;
    excelImportReport = null;
    await selectPatient(currentPatient.id);
}

async function generateCaseReport() {
    if (!currentCaseId) return;
    await runCaseBusyAction('Команда получена, формируется эпикриз...', async () => {
        const result = await apiJson(`/api/cases/${currentCaseId}/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ llm_model: 'qwen2.5:7b-instruct' }),
        });
        if (result.error) { alert(`Ошибка генерации: ${result.error}`); return; }
        currentReportPreview = result;
        currentCaseSubTab = 'reports';
        await refreshActiveCase();
    });
    renderHospitalDashboard();
}

// -------------------- CRUD modals --------------------
async function openObservationModal(category) {
    await ensureCatalog();
    const list = category === 'vital' ? medicalCatalog.vitals : medicalCatalog.labs;
    const options = list.map(item => `<option value="${item.code}">${item.name_ru} (${item.code})</option>`).join('');
    const title = category === 'vital' ? 'Витальный показатель' : 'Лабораторный анализ';
    showFormModal({
        title,
        fields: [
            { name: 'code', label: 'Показатель', input: `<select name="code">${options}</select>` },
            { name: 'value_num', label: 'Значение', type: 'number', step: 'any', required: true },
            { name: 'recorded_at', label: 'Время', type: 'datetime-local', value: nowForInput() },
            { name: 'note', label: 'Комментарий' },
        ],
        onSubmit: async (values) => {
            const url = category === 'vital'
                ? `/api/cases/${currentCaseId}/vitals`
                : `/api/cases/${currentCaseId}/labs`;
            const payload = {
                code: values.code,
                value_num: toNumber(values.value_num),
                recorded_at: values.recorded_at ? new Date(values.recorded_at).toISOString() : null,
                note: values.note || '',
                auto_reassess: true,
            };
            const result = await runCaseBusyAction('Запись сохранена, идет автоматическая переоценка...', async () => {
                const response = await apiJson(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (!response.error) {
                    await refreshActiveCase();
                }
                return response;
            });
            if (!result) return false;
            if (result.error) { alert(result.error); return false; }
            renderHospitalDashboard();
            return true;
        },
    });
}

async function deleteObservation(id, category) {
    if (!confirm('Удалить запись?')) return;
    const url = category === 'vital' ? `/api/vitals/${id}` : `/api/labs/${id}`;
    const result = await apiJson(url, { method: 'DELETE' });
    if (result.error) { alert(result.error); return; }
    await refreshActiveCase();
    renderHospitalDashboard();
}

async function openEntityModal(kind) {
    await ensureCatalog();
    if (kind === 'study') {
        const options = medicalCatalog.studies.map(s => `<option value="${s.code}">${s.name_ru}</option>`).join('');
        showFormModal({
            title: 'Исследование',
            fields: [
                { name: 'code', label: 'Тип', input: `<select name="code">${options}</select>` },
                { name: 'started_at', label: 'Начато', type: 'datetime-local', value: nowForInput() },
                { name: 'completed_at', label: 'Завершено', type: 'datetime-local' },
                { name: 'status', label: 'Статус', input: statusSelect('study') },
                { name: 'result_text', label: 'Результат' },
            ],
            onSubmit: async (v) => submitCrud(`/api/cases/${currentCaseId}/studies`, {
                code: v.code,
                started_at: v.started_at ? new Date(v.started_at).toISOString() : null,
                completed_at: v.completed_at ? new Date(v.completed_at).toISOString() : null,
                status: v.status || 'done',
                result_text: v.result_text || '',
                auto_reassess: true,
            }),
        });
    } else if (kind === 'procedure') {
        const options = medicalCatalog.procedures.map(s => `<option value="${s.code}">${s.name_ru}</option>`).join('');
        showFormModal({
            title: 'Процедура',
            fields: [
                { name: 'code', label: 'Тип', input: `<select name="code">${options}</select>` },
                { name: 'started_at', label: 'Начато', type: 'datetime-local', value: nowForInput() },
                { name: 'completed_at', label: 'Завершено', type: 'datetime-local' },
                { name: 'status', label: 'Статус', input: statusSelect('procedure') },
                { name: 'operator', label: 'Исполнитель' },
                { name: 'note', label: 'Комментарий' },
            ],
            onSubmit: async (v) => submitCrud(`/api/cases/${currentCaseId}/procedures`, {
                code: v.code,
                started_at: v.started_at ? new Date(v.started_at).toISOString() : null,
                completed_at: v.completed_at ? new Date(v.completed_at).toISOString() : null,
                status: v.status || 'done',
                operator: v.operator || '',
                note: v.note || '',
                auto_reassess: true,
            }),
        });
    } else if (kind === 'medication') {
        const options = medicalCatalog.medications.map(m =>
            `<option value="${m.code}" data-class="${m.group}" data-dose="${m.typical_dose}" data-unit="${m.typical_unit}" data-route="${m.default_route}">${m.name_ru}</option>`
        ).join('');
        showFormModal({
            title: 'Назначение',
            fields: [
                { name: 'code', label: 'Препарат', input: `<select name="code">${options}</select>` },
                { name: 'dose', label: 'Доза' },
                { name: 'unit', label: 'Ед.' },
                { name: 'route', label: 'Путь', input: `<select name="route">${['po','iv','iv_drip','sc','im','inhale','sublingual','rectal','topical'].map(r => `<option>${r}</option>`).join('')}</select>` },
                { name: 'frequency', label: 'Кратность', placeholder: 'напр. 1 раз/сут' },
                { name: 'started_at', label: 'Начато', type: 'datetime-local', value: nowForInput() },
                { name: 'status', label: 'Статус', input: statusSelect('medication') },
            ],
            onSubmit: async (v) => submitCrud(`/api/cases/${currentCaseId}/medications`, {
                code: v.code,
                dose: v.dose,
                unit: v.unit,
                route: v.route,
                frequency: v.frequency,
                started_at: v.started_at ? new Date(v.started_at).toISOString() : null,
                status: v.status || 'active',
                auto_reassess: true,
            }),
        });
    } else if (kind === 'diagnosis') {
        const options = medicalCatalog.diagnoses.map(d => `<option value="${d.icd10}">${d.icd10} — ${d.name_ru}</option>`).join('');
        showFormModal({
            title: 'Диагноз',
            fields: [
                { name: 'icd10', label: 'МКБ-10', input: `<select name="icd10">${options}</select>` },
                { name: 'diagnosis_type', label: 'Тип', input: `<select name="diagnosis_type">${['primary','secondary','complication'].map(s => `<option>${s}</option>`).join('')}</select>` },
                { name: 'established_at', label: 'Установлен', type: 'datetime-local', value: nowForInput() },
                { name: 'note', label: 'Комментарий' },
            ],
            onSubmit: async (v) => submitCrud(`/api/cases/${currentCaseId}/diagnoses`, {
                icd10: v.icd10,
                diagnosis_type: v.diagnosis_type || 'primary',
                established_at: v.established_at ? new Date(v.established_at).toISOString() : null,
                note: v.note || '',
                auto_reassess: true,
            }),
        });
    }
}

async function submitCrud(url, payload) {
    const result = await runCaseBusyAction('Запись сохранена, идет автоматическая переоценка...', async () => {
        const response = await apiJson(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.error) {
            await refreshActiveCase();
        }
        return response;
    });
    if (!result) return false;
    if (result.error) { alert(result.error); return false; }
    renderHospitalDashboard();
    return true;
}

async function deleteEntity(kind, id) {
    if (!confirm('Удалить запись?')) return;
    const map = { study: 'studies', procedure: 'procedures', medication: 'medications', diagnosis: 'diagnoses' };
    const url = `/api/${map[kind]}/${id}`;
    const result = await apiJson(url, { method: 'DELETE' });
    if (result.error) { alert(result.error); return; }
    await refreshActiveCase();
    renderHospitalDashboard();
}

async function uploadExcel(dryRun) {
    const fileInput = document.getElementById('excelFile');
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        alert('Выберите .xlsx файл');
        return;
    }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    const url = `/api/cases/${currentCaseId}/excel-import?dry_run=${dryRun ? 'true' : 'false'}`;
    await runCaseBusyAction(
        dryRun ? 'Проверяем Excel-файл...' : 'Импортируем Excel и обновляем оценку...',
        async () => {
            const response = await fetch(url, { method: 'POST', body: fd });
            const result = await response.json();
            excelImportReport = result;
            if (!dryRun && !result.error) {
                await refreshActiveCase();
            }
        }
    );
    renderHospitalDashboard();
}

// -------------------- Generic modal --------------------
function showFormModal({ title, fields, onSubmit }) {
    const existing = document.getElementById('dynamicModal');
    if (existing) existing.remove();

    const fieldsHtml = fields.map(f => {
        if (f.input) return `<label>${escapeHtml(f.label)}</label>${f.input}`;
        const attrs = [
            `name="${f.name}"`,
            `type="${f.type || 'text'}"`,
            f.step ? `step="${f.step}"` : '',
            f.required ? 'required' : '',
            f.placeholder ? `placeholder="${escapeHtml(f.placeholder)}"` : '',
            f.value ? `value="${escapeHtml(f.value)}"` : '',
        ].filter(Boolean).join(' ');
        return `<label>${escapeHtml(f.label)}</label><input ${attrs}>`;
    }).join('');

    const modal = document.createElement('div');
    modal.id = 'dynamicModal';
    modal.className = 'modal';
    modal.style.display = 'flex';
    modal.innerHTML = `
        <div class="modal-content" style="width: 480px;">
            <span class="close-btn" onclick="document.getElementById('dynamicModal').remove()">&times;</span>
            <h3 style="margin-top:0;">${escapeHtml(title)}</h3>
            <form id="dynamicForm" style="display:flex; flex-direction:column; gap:8px;">
                ${fieldsHtml}
                <button type="submit" class="action-btn" style="margin-top: 10px;">Сохранить</button>
            </form>
        </div>
    `;
    document.body.appendChild(modal);
    const form = document.getElementById('dynamicForm');
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const data = {};
        new FormData(form).forEach((value, key) => { data[key] = value; });
        const ok = await onSubmit(data);
        if (ok !== false) {
            modal.remove();
        }
    });
}

function statusSelect(kind) {
    const options = {
        study: ['ordered', 'in_progress', 'done', 'cancelled'],
        procedure: ['ordered', 'in_progress', 'done', 'cancelled'],
        medication: ['active', 'paused', 'stopped', 'completed'],
    }[kind] || ['ordered'];
    return `<select name="status">${options.map(o => `<option>${o}</option>`).join('')}</select>`;
}

function statusClass(status) {
    if (!status) return 'neutral';
    if (['done', 'completed', 'active'].includes(status)) return 'active';
    if (['ordered', 'in_progress', 'paused'].includes(status)) return 'warn';
    if (['cancelled', 'stopped'].includes(status)) return 'done';
    return 'neutral';
}

function flagLabel(flag) {
    const map = { norm: 'норма', low: 'ниже нормы', high: 'выше нормы', critical_low: 'критически низко', critical_high: 'критически высоко', unknown: '—' };
    return map[flag] || '—';
}

function rangeText(low, high) {
    if (low == null && high == null) return '';
    return `${low ?? ''}–${high ?? ''}`.replace(/^–$/, '');
}

function fmtNum(v) {
    if (v === null || v === undefined) return '';
    return String(v);
}

function formatDt(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    return d.toLocaleString();
}

function nowForInput() {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    return now.toISOString().slice(0, 16);
}

function toNumber(value) {
    if (value === '' || value === undefined || value === null) return null;
    const n = parseFloat(String(value).replace(',', '.'));
    return isNaN(n) ? null : n;
}


// Логика модального окна
function openVisitModal() {
    if (!currentPatient) return;
    
    document.getElementById('modalPatientId').value = currentPatient.id;
    document.getElementById('modalPatientName').value = currentPatient.full_name;
    
    // Подставляем текущую дату и время 
    // slice(0,16) обрезает строку до формата YYYY-MM-DDTHH:MM (то, что нужно для datetime-local)
    const now = new Date();

    // Корректируем время под локальный часовой пояс пользователя
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    document.getElementById('modalVisitDate').value = now.toISOString().slice(0,16);
    
    document.getElementById('visitModal').style.display = 'flex';
}

function closeVisitModal() {
    // Скрываем окно
    document.getElementById('visitModal').style.display = 'none';
}

async function submitNewVisit() {
    const dateVal = document.getElementById('modalVisitDate').value;
    if(!dateVal) {
        alert("Укажите дату и время визита!");
        return;
    }

    // Проверка частого создания записей (в пределах одного часа на человека)
    // Переводим введенное время в миллисекунды
    const newTimeMs = new Date(dateVal).getTime();
    
    // Ищем, есть ли визит с разницей меньше 60 минут (1 час = 3 600 000 мс)
    const isTooClose = currentPatient.visits.some(v => {
        const existingTimeMs = new Date(v.iso_date).getTime();
        const diffHours = Math.abs(newTimeMs - existingTimeMs) / (1000 * 60 * 60);
        return diffHours <= 1.0;
    });

    if (isTooClose) {
        // confirm выводит стандартное окно с кнопками "ОК" и "Отмена"
        const userAgreed = confirm("⚠️ Внимание!\nУ пациента уже есть визит с разницей менее 1 часа от указанного времени.\nВы точно хотите добавить еще один?");
        if (!userAgreed) {
            return;
        }
    }

    try {
        const response = await fetch('/api/visits', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                patient_id: currentPatient.id,
                date: dateVal
            })
        });
        
        const res = await response.json();
        if(res.error) throw new Error(res.error);
        
        closeVisitModal();
        selectPatient(currentPatient.id); // получаем обновленные данные пациента
        
    } catch (err) {
        alert("Ошибка: " + err.message);
    }
}

// Удаление визита
async function deleteVisit(visitId) {
    // Спрашиваем подтверждение перед удалением
    if (!confirm("Вы уверены, что хотите удалить этот визит? Это действие необратимо.")) {
        return;
    }

    try {
        const response = await fetch(`/api/visits/${visitId}`, {
            method: 'DELETE'
        });
        
        const res = await response.json();
        if(res.error) throw new Error(res.error);
        
        // Перезагружаем пациента, чтобы визит исчез с экрана
        selectPatient(currentPatient.id); 
        
    } catch (err) {
        alert("Ошибка удаления: " + err.message);
    }
}

// Добавление нового пациента
function openAddPatientModal() {
    // Очищаем поля перед открытием
    document.getElementById('newLastName').value = '';
    document.getElementById('newFirstName').value = '';
    document.getElementById('newPatronymic').value = '';
    document.getElementById('newBirthDate').value = '';
    document.getElementById('newGender').value = '';
    
    document.getElementById('patientModal').style.display = 'flex';
}

function closeAddPatientModal() {
    document.getElementById('patientModal').style.display = 'none';
}

async function submitNewPatient() {
    // Считываем данные
    const lastName = document.getElementById('newLastName').value.trim();
    const firstName = document.getElementById('newFirstName').value.trim();
    const patronymic = document.getElementById('newPatronymic').value.trim();
    const birthDate = document.getElementById('newBirthDate').value;
    const gender = document.getElementById('newGender').value;

    // Проверка обязательный полей
    if (!lastName || !firstName || !birthDate || !gender) {
        alert("Пожалуйста, заполните все обязательные поля (отмечены звездочкой *)!");
        return; // Прерываем выполнение! Данные не уйдут на сервер.
    }

    try {
        // Отправляем запрос
        const response = await fetch('/api/patients', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                last_name: lastName,
                first_name: firstName,
                patronymic: patronymic,
                birth_date: birthDate,
                gender: gender
            })
        });
        
        const result = await response.json();
        
        if (result.error) throw new Error(result.error);
        
        alert(`Пациент успешно зарегистрирован!\nПрисвоен ID: ${result.display_id}`);
        closeAddPatientModal();
        
        // Перезагружаем список пациентов слева
        await loadPatientsFromDB();
        
    } catch (err) {
        alert("Ошибка при регистрации: " + err.message);
    }
}


// Логика панели настроек
let currentNsiMode = 'edit';

// Открываем таблицу (Справочник)
async function openNsiModal(mode) {
    currentNsiMode = mode;
    const modal = document.getElementById('nsiModal');
    const title = document.getElementById('nsiModalTitle');
    const hint = document.getElementById('nsiModalHint');
    const tbody = document.getElementById('nsiTableBody');
    
    // Показываем окно и крутилку загрузки
    modal.style.display = 'flex';
    tbody.innerHTML = "<tr><td colspan='4' style='padding:20px; text-align:center; color:#94a3b8;'>Загрузка справочника из базы данных...</td></tr>";
    
    // Настраиваем заголовки
    if (mode === 'edit') {
        title.innerText = "Редактирование пациента";
        hint.innerText = "Выберите пациента для изменения его данных.";
    } else {
        title.innerText = "Удаление пациента";
        hint.innerHTML = "<b style='color:#ef4444;'>ВНИМАНИЕ!</b> Удаление пациента приведет к безвозвратной потере всех его визитов и анализов.";
    }

    try {
        // Запрос пациентов
        const response = await fetch('/api/patients');
        const patientsList = await response.json();

        // Очищаем таблицу и рисуем результат
        tbody.innerHTML = "";
        
        if (patientsList.length === 0) {
            tbody.innerHTML = "<tr><td colspan='4' style='padding:20px; text-align:center; color:#94a3b8;'>В базе данных нет пациентов.</td></tr>";
            return;
        }

        patientsList.forEach(p => {
            // Формируем кнопку действия
            let actionBtn = "";
            if (mode === 'edit') {
                // Важно: мы передаем ID в openEditForm, а не весь объект!
                actionBtn = `<button class="small-btn" style="color:#2563eb; border-color:#2563eb;" onclick="openEditForm(${p.id})">Редактировать</button>`;
            } else {
                // Внешние одинарные кавычки: иначе JSON.stringify даёт "…" и рвёт onclick="…"
                actionBtn = `<button type="button" class="small-btn" style="color:#ef4444; border-color:#ef4444; background-color:#fef2f2;" onclick='confirmDelete(${p.id}, ${JSON.stringify(p.full_name)})'>Удалить</button>`;
            }

            // Рисуем строку таблицы
            tbody.innerHTML += `
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px; font-family: monospace; color: #3b82f6; font-weight: bold;">${p.display_id || p.id}</td>
                    <td style="padding: 10px; font-weight: 500; color: #1e293b;">${p.full_name}</td>
                    <td style="padding: 10px; color: #475569;">${p.birth_date}</td>
                    <td style="padding: 10px; text-align: right;">${actionBtn}</td>
                </tr>
            `;
        });

    } catch (err) {
        tbody.innerHTML = `<tr><td colspan='4' style='padding:20px; text-align:center; color:red;'>Ошибка загрузки: ${err.message}</td></tr>`;
    }
}

// Логика удаления
async function confirmDelete(patientId, patientName) {
    // Двойное жесткое подтверждение от пользователя
    const isConfirmed = confirm(
        `Безвозвратное удаление!\n\nВы действительно хотите удалить пациента "${patientName}" и всю его историю визитов?`
    );
    
    if (!isConfirmed) return;

    try {
        const response = await fetch(`/api/patients/${patientId}`, { method: 'DELETE' });
        let result = {};
        try {
            result = await response.json();
        } catch (_) {
            throw new Error(`Сервер ответил ${response.status} (не JSON)`);
        }
        if (!response.ok) throw new Error(result.error || result.detail || `HTTP ${response.status}`);
        if (result.error) throw new Error(result.error);
        
        alert("Пациент успешно удален.");
        document.getElementById('nsiModal').style.display = 'none';
        
        // Сбрасываем выбранного пациента, если мы удалили именно его
        if (currentPatient && currentPatient.id === patientId) {
            unselectPatient();
        }
        
        // Обновляем список слева
        await loadPatientsFromDB();
        
    } catch (err) {
        alert("Ошибка удаления: " + err.message);
    }
}

// Логика редактирования пациента
async function openEditForm(patientId) {
    try {
        // Запрашиваем полные данные пациента по ID
        const response = await fetch(`/api/patients/${patientId}`);
        const p = await response.json();

        if (p.error) throw new Error(p.error);

        // акрываем таблицу Справочника
        document.getElementById('nsiModal').style.display = 'none';
        
        // Разбиваем ФИО обратно на части
        const nameParts = p.full_name.split(' ');
        const lastName = nameParts[0] || '';
        const firstName = nameParts[1] || '';
        const patronymic = nameParts.slice(2).join(' ') || '';

        // Переворачиваем дату из ДД.ММ.ГГГГ (как в базе) в ГГГГ-ММ-ДД (для HTML-календаря)
        const dateParts = p.birth_date.split('.');
        const isoDate = `${dateParts[2]}-${dateParts[1]}-${dateParts[0]}`;

        // Заполняем поля старой формы "Добавить пациента"
        document.getElementById('newLastName').value = lastName;
        document.getElementById('newFirstName').value = firstName;
        document.getElementById('newPatronymic').value = patronymic;
        document.getElementById('newBirthDate').value = isoDate;
        document.getElementById('newGender').value = p.gender || '';
        
        // Меняем надпись на кнопке и сохраняем ID пациента в скрытый атрибут
        const btn = document.querySelector('#patientModal .action-btn');
        btn.innerText = "Сохранить изменения";
        btn.setAttribute('data-edit-id', p.id); 
        
        // Открываем окно
        document.getElementById('patientModal').style.display = 'flex';

    } catch (err) {
        alert("Ошибка при получении данных пациента: " + err.message);
    }
}

// Обновляем старую функцию submitNewPatient, чтобы она умела и создавать, и обновлять
async function submitNewPatient() {
    const lastName = document.getElementById('newLastName').value.trim();
    const firstName = document.getElementById('newFirstName').value.trim();
    const patronymic = document.getElementById('newPatronymic').value.trim();
    const birthDate = document.getElementById('newBirthDate').value;
    const gender = document.getElementById('newGender').value;

    if (!lastName || !firstName || !birthDate || !gender) {
        alert("Заполните обязательные поля!"); return;
    }

    // Проверяем, создаем мы нового или редактируем старого
    const btn = document.querySelector('#patientModal .action-btn');
    const editId = btn.getAttribute('data-edit-id');
    
    try {
        let url = '/api/patients';
        let method = 'POST';
        
        if (editId) {
            url = `/api/patients/${editId}`;
            method = 'PUT'; // Обновление
        }

        const response = await fetch(url, {
            method: method,
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                last_name: lastName, first_name: firstName, patronymic: patronymic,
                birth_date: birthDate, gender: gender
            })
        });
        
        const result = await response.json();
        if (result.error) throw new Error(result.error);
        
        alert(editId ? "Данные успешно обновлены!" : `Пациент зарегистрирован!\nID: ${result.display_id}`);
        closeAddPatientModal();
        
        // Сбрасываем атрибут кнопки обратно
        btn.innerText = "Зарегистрировать";
        btn.removeAttribute('data-edit-id');
        
        // Перезагружаем список
        await loadPatientsFromDB();
        
        // Если мы редактировали текущего пациента, обновляем его панель
        if (editId && currentPatient && currentPatient.id == editId) {
            selectPatient(currentPatient.id);
        }
        
    } catch (err) {
        alert("Ошибка: " + err.message);
    }
}