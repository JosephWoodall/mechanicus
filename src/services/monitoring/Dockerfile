FROM python:3.9-slim

# Install Docker CLI and docker-compose
RUN apt-get update && apt-get install -y \
    docker.io \
    docker-compose \
    && rm -rf /var/lib/apt/lists/*

# Add user to docker group (optional)
RUN groupadd -g 999 docker || true
RUN usermod -a -G docker root || true

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python", "redis_channel_monitor.py"]