FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/shared/config /app/shared/data /app/shared/models

COPY src/ml_training_dataset_generator.py .
COPY src/ml_training.py .
COPY src/rl_training.py .

CMD ["python", "ml_training_dataset_generator.py", "--config", "/app/shared/config/mechanicus_run_configuration.yaml"]