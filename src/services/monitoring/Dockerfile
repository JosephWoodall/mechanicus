FROM python:3.9-slim

# Install Docker CLI
RUN apt-get update && apt-get install -y \
    docker.io \
    docker-compose \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add docker group and user
RUN groupadd -g 999 docker || true
RUN usermod -a -G docker root || true

WORKDIR /app
COPY src/ .

CMD ["python", "redis_channel_monitor.py"]