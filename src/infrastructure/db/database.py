# src/infrastructure/db/database.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

""" Создаем сеанс связи для общения с базой данных (postgresql) """

# Парсим DATABASE_URL из файла .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL не найден в файле .env")

# Создаем движок
engine = create_engine(DATABASE_URL,
                       echo=False # echo=True покажет все SQL запросы в консоли
                       )

# Фабрика сессий для работы с БД
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для всех таблиц
class Base(DeclarativeBase):
    pass
