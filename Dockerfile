# Dockerfile
FROM apache/airflow:2.7.2-python3.10

# Установка зависимостей
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /requirements.txt

# Копируем твой проект (скрипты)
COPY src/ /opt/airflow/src/
