FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create shared directories
RUN mkdir -p /app/shared/config /app/shared/data

# Copy the training dataset generator
COPY src/ml_training_dataset_generator.py .

# Default command to generate training data
CMD ["python", "ml_training_dataset_generator.py", "--config", "/app/shared/config/mechanicus_run_configuration.yaml"]
