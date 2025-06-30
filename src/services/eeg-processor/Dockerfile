FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./

RUN mkdir -p /app/shared/models

CMD ["python", "utils/preprocessing.py"]