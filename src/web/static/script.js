let currentPatient = null; // Здесь будет храниться объект выбранного пациента

// Логика переключения вкладок
function switchTab(tabId, btnElement) {

    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    btnElement.classList.add('active');
}

// Логика поиска пациентов
function searchPatient() {

    const rawInput = document.getElementById('searchInput').value.trim();
    const searchMessage = document.getElementById('searchMessage');
    
    const cards = document.querySelectorAll('.patient-card');

    if (rawInput === "") {
        cards.forEach(card => card.style.display = "block");
        searchMessage.style.display = "none";
        return; // Завершаем функцию
    }

    const query = rawInput.toLowerCase();
    
    const isIdSearch = /^\d$/.test(query.charAt(0)); 
    
    let matchCount = 0; // Счетчик найденных пациентов

    cards.forEach(card => {

        const idElement = card.querySelector('.patient-card-header span:first-child');
        const nameElement = card.querySelector('.patient-card-name');
        
        let matchFound = false;

        if (isIdSearch) {
            const rawIdText = idElement.textContent; // "ID: 10042"
            const idNumber = rawIdText.replace(/\D/g, ''); // Удаляем всё, кроме цифр -> "10042"
            
            if (idNumber.startsWith(query)) {
                matchFound = true;
            }
        } else {
            const nameText = nameElement.textContent.toLowerCase();
            if (nameText.includes(query)) {
                matchFound = true;
            }
        }

        if (matchFound) {
            card.style.display = "block";
            matchCount++;
        } else {
            card.style.display = "none";
        }
    });

    if (matchCount === 0) {
        // Никого не нашли
        const searchType = isIdSearch ? "ID" : "ФИО";
        searchMessage.textContent = `Пациент с ${searchType} "${rawInput}" не найден.`;
        searchMessage.style.display = "block";
    } else {
        // Нашли кого-то, прячем сообщение об ошибке
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
async function runAssessment() {
    const outputBox = document.getElementById('resultOutput');
    outputBox.innerText = "Загрузка..."; // Крутилка

    const payload = {
        name: document.getElementById('ptName').value,
        pain_type: document.getElementById('ptPain').value,
        ecg_changes: document.getElementById('ptEcg').value,
        troponin: parseFloat(document.getElementById('ptTrop').value),
        hr: parseInt(document.getElementById('ptHr').value),
        bp: document.getElementById('ptBp').value,
        free_text: ""
    };

    const response = await fetch('/api/assess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    const result = await response.json();
    outputBox.innerText = JSON.stringify(result, null, 2);
}

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

document.addEventListener('DOMContentLoaded', async function () {
    const resizer = document.getElementById('dragMe');
    const leftSide = document.getElementById('sidebar');

    let x = 0;
    let leftWidth = 0;

    const mouseDownHandler = function (e) {
        x = e.clientX;
        leftWidth = leftSide.getBoundingClientRect().width;

        document.addEventListener('mousemove', mouseMoveHandler);
        document.addEventListener('mouseup', mouseUpHandler);
        
        resizer.classList.add('active');
        document.body.style.cursor = 'col-resize';
        
        leftSide.style.userSelect = 'none';
        leftSide.style.pointerEvents = 'none';
    };

    const mouseMoveHandler = function (e) {
        const dx = e.clientX - x; // На сколько пикселей сдвинули мышь
        const newLeftWidth = ((leftWidth + dx) * 100) / resizer.parentNode.getBoundingClientRect().width;
        
        leftSide.style.width = `${newLeftWidth}%`;
    };

    const mouseUpHandler = function () {
        resizer.classList.remove('active');
        document.body.style.cursor = 'default';
        leftSide.style.userSelect = 'auto';
        leftSide.style.pointerEvents = 'auto';
        
        document.removeEventListener('mousemove', mouseMoveHandler);
        document.removeEventListener('mouseup', mouseUpHandler);
    };

    resizer.addEventListener('mousedown', mouseDownHandler);

    // Загружаем пациентов
    await loadPatientsFromDB();
});

async function loadPatientsFromDB() {
    const listContainer = document.getElementById('patientList');
    listContainer.innerHTML = "<p style='text-align:center; color:#94a3b8;'>Загрузка...</p>";

    try {
        // Делаем GET запрос к нашему Python API
        const response = await fetch('/api/patients');
        const patients = await response.json();

        // Очищаем контейнер
        listContainer.innerHTML = "";

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


// логика выбора пациента
async function selectPatient(patientId) {
    document.getElementById('visitsPanel').innerHTML = "Загрузка данных пациента...";
    
    try {
        const response = await fetch(`/api/patients/${patientId}`);
        const patient = await response.json();

        if (patient.error) throw new Error(patient.error);

        currentPatient = patient;

        updateSelectionUI(patientId);

        renderPatientDashboard();

    } catch (error) {
        console.error("Ошибка при получении деталей:", error);
        alert("Не удалось загрузить данные пациента");
    }
}

// Вынесли визуальное обновление в отдельную функцию для чистоты
function updateSelectionUI(patientId) {
    document.querySelectorAll('.patient-card').forEach(card => card.classList.remove('selected'));
    const activeCard = document.getElementById(`card-${patientId}`);
    if (activeCard) activeCard.classList.add('selected');

    document.getElementById('activePatientText').innerText = 
        `Выбран пациент: [ID: ${currentPatient.id}] ${currentPatient.full_name}`;
    document.getElementById('unselectBtn').style.display = 'block';
}

function renderPatientDashboard() {
    const visitsPanel = document.getElementById('visitsPanel');
    const statsPanel = document.getElementById('statsPanel');

    if (!currentPatient) {
        visitsPanel.innerHTML = '<div class="empty-state">Выберите пациента</div>';
        statsPanel.innerHTML = '<div class="empty-state">Выберите пациента</div>';
        return;
    }

    // Рендерим визиты, которые мы только что получили из БД
    let visitsHTML = `<h4>История визитов: ${currentPatient.full_name}</h4>`;
    
    if (currentPatient.visits.length === 0) {
        visitsHTML += "<p>Визитов пока нет</p>";
    } else {
        currentPatient.visits.forEach(v => {
            visitsHTML += `
                <div class="visit-item" style="border-bottom: 1px solid #eee; padding: 10px 0;">
                    <b>Дата:</b> ${v.date} <br>
                    <b>Риск:</b> ${v.risk}
                </div>`;
        });
    }

    visitsHTML += `<br><button class="action-btn" style="width:100%">+ Новая госпитализация</button>`;
    
    visitsPanel.innerHTML = visitsHTML;
    statsPanel.innerHTML = "<h4>Статистика и анализы</h4><p>Данные подгружены из БД.</p>";
}

function unselectPatient() {
    // Очищаем состояние
    currentPatient = null;

    // Убираем подсветку
    document.querySelectorAll('.patient-card').forEach(card => card.classList.remove('selected'));

    // Сбрасываем плашку
    document.getElementById('activePatientText').innerText = 'Пациент не выбран';
    document.getElementById('unselectBtn').style.display = 'none';

    // Очищаем вкладку "Пациент"
    renderPatientDashboard();
}

function renderPatientDashboard() {
    const visitsPanel = document.getElementById('visitsPanel');
    const statsPanel = document.getElementById('statsPanel');

    // 1. Если пациент не выбран - показываем пустые экраны (заглушки)
    if (!currentPatient) {
        visitsPanel.innerHTML = '<div class="empty-state">Выберите пациента слева для просмотра визитов</div>';
        statsPanel.innerHTML = '<div class="empty-state">Выберите пациента слева для просмотра анализов</div>';
        return;
    }


    // Формируем заголовок колонки визитов с маленькой кнопкой "+ Добавить"
    let visitsHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; margin-bottom: 15px;">
            <h4 style="margin: 0; color: #334155;">История визитов</h4>
            <button class="small-btn" onclick="openVisitModal()">+ Добавить</button>
        </div>
    `;
    
    // Если у пациента в базе нет ни одного визита
    if (!currentPatient.visits || currentPatient.visits.length === 0) {
        visitsHTML += "<p style='color: #94a3b8; font-style: italic; text-align: center; margin-top: 30px;'>Визитов пока нет</p>";
    } 
    // Если визиты есть, рисуем список
    else {
        // Сортируем визиты по дате (чтобы новые были сверху)
        const sortedVisits = [...currentPatient.visits].reverse();
        
        sortedVisits.forEach(v => {
            visitsHTML += `
                <div class="visit-item" style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding: 12px 0; color: #475569;">
                    <span style="font-size: 1.1em; font-weight: 500;">📅 ${v.date}</span>
                    <button onclick="deleteVisit(${v.id})" style="background: none; border: none; color: #ef4444; cursor: pointer; font-size: 1.2em;" title="Удалить визит">🗑️</button>
                </div>
            `;
        });
    }
    
    // Вставляем сгенерированный HTML в левую панель вкладки "Пациент"
    visitsPanel.innerHTML = visitsHTML;

    
    // Пока что просто рисуем заглушку, так как анализы мы еще не делали
    statsPanel.innerHTML = `
        <div style="border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; margin-bottom: 15px;">
            <h4 style="margin: 0; color: #334155;">Анализы и статистика</h4>
        </div>
        
        <div style="color: #64748b; line-height: 1.6;">
            <p>Здесь будут выводиться графики тропонина и результаты расшифровки ЭКГ для пациента <b>${currentPatient.full_name}</b>.</p>
            
            <div style="background-color: #f8f9fa; border: 1px dashed #cbd5e1; border-radius: 6px; padding: 20px; text-align: center; margin-top: 30px;">
                <p style="margin: 0; font-style: italic;">(Раздел анализов в разработке...)</p>
            </div>
        </div>
    `;
}

// логика модального окна визитов
function openVisitModal() {
    if (!currentPatient) return;
    
    document.getElementById('modalPatientId').value = currentPatient.id;
    document.getElementById('modalPatientName').value = currentPatient.full_name;
    
    const now = new Date();
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

    const newTimeMs = new Date(dateVal).getTime();
    
    const isTooClose = currentPatient.visits.some(v => {
        const existingTimeMs = new Date(v.iso_date).getTime();
        const diffHours = Math.abs(newTimeMs - existingTimeMs) / (1000 * 60 * 60);
        return diffHours <= 1.0;
    });

    if (isTooClose) {
        const userAgreed = confirm("⚠️ Внимание!\nУ пациента уже есть визит с разницей менее 1 часа от указанного времени.\nВы точно хотите добавить еще один?");
        if (!userAgreed) {
            return; // Пользователь нажал "Отмена" - прерываем сохранение
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
        selectPatient(currentPatient.id); 
        
    } catch (err) {
        alert("Ошибка: " + err.message);
    }
}

// Удаление визита
async function deleteVisit(visitId) {
    // Спрашиваем подтверждение перед удалением (защита от случайного клика)
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
    // 1. Считываем данные
    const lastName = document.getElementById('newLastName').value.trim();
    const firstName = document.getElementById('newFirstName').value.trim();
    const patronymic = document.getElementById('newPatronymic').value.trim();
    const birthDate = document.getElementById('newBirthDate').value;
    const gender = document.getElementById('newGender').value;

    if (!lastName || !firstName || !birthDate || !gender) {
        alert("Пожалуйста, заполните все обязательные поля (отмечены звездочкой *)!");
        return;
    }

    try {
        // 3. Отправляем запрос
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
        
        await loadPatientsFromDB();
        
    } catch (err) {
        alert("Ошибка при регистрации: " + err.message);
    }
}


// НСИ
let currentNsiMode = 'edit';

// Открываем таблицу (Справочник)
async function openNsiModal(mode) {
    currentNsiMode = mode;
    const modal = document.getElementById('nsiModal');
    const title = document.getElementById('nsiModalTitle');
    const hint = document.getElementById('nsiModalHint');
    const tbody = document.getElementById('nsiTableBody');
    
    modal.style.display = 'flex';
    tbody.innerHTML = "<tr><td colspan='4' style='padding:20px; text-align:center; color:#94a3b8;'>Загрузка справочника из базы данных...</td></tr>";
    
    if (mode === 'edit') {
        title.innerText = "Редактирование пациента";
        hint.innerText = "Выберите пациента для изменения его данных.";
    } else {
        title.innerText = "Удаление пациента";
        hint.innerHTML = "<b style='color:#ef4444;'>ВНИМАНИЕ!</b> Удаление пациента приведет к безвозвратной потере всех его визитов и анализов.";
    }

    try {
        const response = await fetch('/api/patients');
        const patientsList = await response.json();

        tbody.innerHTML = "";
        
        if (patientsList.length === 0) {
            tbody.innerHTML = "<tr><td colspan='4' style='padding:20px; text-align:center; color:#94a3b8;'>В базе данных нет пациентов.</td></tr>";
            return;
        }

        patientsList.forEach(p => {
            let actionBtn = "";
            if (mode === 'edit') {

                actionBtn = `<button class="small-btn" style="color:#2563eb; border-color:#2563eb;" onclick="openEditForm(${p.id})">Редактировать</button>`;
            } else {
                actionBtn = `<button class="small-btn" style="color:#ef4444; border-color:#ef4444; background-color:#fef2f2;" onclick="confirmDelete(${p.id}, '${p.full_name}')">Удалить</button>`;
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

// Удаление
async function confirmDelete(patientId, patientName) {
    // Двойное жесткое подтверждение от пользователя
    const isConfirmed = confirm(
        `БЕЗВОЗВРАТНОЕ УДАЛЕНИЕ!\n\nВы действительно хотите удалить пациента "${patientName}" и всю его историю визитов?`
    );
    
    if (!isConfirmed) return;

    try {
        const response = await fetch(`/api/patients/${patientId}`, { method: 'DELETE' });
        const result = await response.json();
        
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

// Редактирование
async function openEditForm(patientId) {
    try {
        const response = await fetch(`/api/patients/${patientId}`);
        const p = await response.json();

        if (p.error) throw new Error(p.error);

        document.getElementById('nsiModal').style.display = 'none';
        
        const nameParts = p.full_name.split(' ');
        const lastName = nameParts[0] || '';
        const firstName = nameParts[1] || '';
        const patronymic = nameParts.slice(2).join(' ') || '';

        const dateParts = p.birth_date.split('.');
        const isoDate = `${dateParts[2]}-${dateParts[1]}-${dateParts[0]}`;

        document.getElementById('newLastName').value = lastName;
        document.getElementById('newFirstName').value = firstName;
        document.getElementById('newPatronymic').value = patronymic;
        document.getElementById('newBirthDate').value = isoDate;
        
        document.getElementById('newGender').value = p.gender || '';
        
        const btn = document.querySelector('#patientModal .action-btn');
        btn.innerText = "Сохранить изменения";
        btn.setAttribute('data-edit-id', p.id); 
        
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