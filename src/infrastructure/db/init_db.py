from src.infrastructure.db.database import engine, Base, SessionLocal
from src.infrastructure.db.models import Patient, Visit
from datetime import date 

""" Скрипт начальной инициализации. Создает все необходимые таблицы в базе """

def init_database():
    print("Создание таблиц в базе ...")
    Base.metadata.create_all(bind=engine)
    
    session = SessionLocal()
    
    # Проверяем, есть ли уже пациенты
    if session.query(Patient).count() == 0:
        print("База пуста. Добавляем тестовых пациентов...")
        p1 = Patient(full_name="Иванов Петр Сергеевич", birth_date=date(1957, 5, 12), gender="male")
        p2 = Patient(full_name="Смирнова Анна Викторовна", birth_date=date(1982, 11, 30), gender="female")
        p3 = Patient(full_name="Кузнецов Игорь Николаевич", birth_date=date(1995, 2, 15), gender="male")
        
        session.add_all([p1, p2, p3])
        session.commit()
        print("Пациенты успешно добавлены!")
    else:
        print("База уже содержит данные.")
        
    session.close()

if __name__ == "__main__":
    init_database()
