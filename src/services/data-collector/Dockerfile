FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/shared/config

COPY src/data_collection.py .

CMD ["python", "data_collection.py", "--config", "/app/shared/config/mechanicus_run_configuration.yaml"]