# Используем базовый образ с Python
FROM python:3.9-slim

# Устанавливаем необходимые пакеты
RUN pip install flask torch

# Копируем файлы приложения и модели
COPY app.py /app/app.py
COPY model.pth /app/model.pth

# Переходим в директорию приложения
WORKDIR /app

# Открываем порт для Flask
EXPOSE 5000

# Запускаем приложение
CMD ["python", "app.py"]
