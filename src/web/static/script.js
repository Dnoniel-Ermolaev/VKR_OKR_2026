// Тип Object - изменяется при выборе пациента.
// Хранит базовую информацию, требуемую для отображения на форме 
let currentPatient = null;

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
async function runAssessment() {
    const outputBox = document.getElementById('resultOutput');
    outputBox.innerText = "Загрузка...";

    // Собираем данные из полей ввода
    const payload = {
        name: document.getElementById('ptName').value,
        pain_type: document.getElementById('ptPain').value,
        ecg_changes: document.getElementById('ptEcg').value,
        troponin: parseFloat(document.getElementById('ptTrop').value),
        hr: parseInt(document.getElementById('ptHr').value),
        bp: document.getElementById('ptBp').value,
        free_text: ""
    };

    // Отправляем JSON на наш Python сервер (FastAPI)
    const response = await fetch('/api/assess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    // Печатаем ответ
    const result = await response.json();
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
    // Показываем загрузку на вкладке
    document.getElementById('visitsPanel').innerHTML = "Загрузка данных пациента...";
    
    try {
        // Делаем запрос к серверу за полными данными
        const response = await fetch(`/api/patients/${patientId}`);
        const patient = await response.json();

        if (patient.error) throw new Error(patient.error);

        // охраняем выбранного пациента в глобальное состояние
        currentPatient = patient;

        // Обновляем интерфейс
        updateSelectionUI(patientId);

        // Отрисовываем детали (визиты и т.д.)
        renderPatientDashboard();

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

// Отрисовываем детали
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

 // Вывести приложение из режима работы с конкретным человеком и вернуть его в нейтральное состояние
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

    // Если пациент не выбран - показываем пустые экраны (заглушки)
    if (!currentPatient) {
        visitsPanel.innerHTML = '<div class="empty-state">Выберите пациента для просмотра визитов</div>';
        statsPanel.innerHTML = '<div class="empty-state">Выберите пациента для просмотра анализов</div>';
        return;
    }

    // 1. Левая колонка (Визиты)

    // Формируем заголовок колонки визитов с кнопкой "Добавить"
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
        // Сортируем визиты по дате
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


    // 2. Правая колонка 
    
    // Пока что просто рисуем заглушку
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

// Логика удаления
async function confirmDelete(patientId, patientName) {
    // Двойное жесткое подтверждение от пользователя
    const isConfirmed = confirm(
        `Безвозвратное удаление!\n\nВы действительно хотите удалить пациента "${patientName}" и всю его историю визитов?`
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