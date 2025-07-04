FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the monitoring script from src subdirectory
COPY src/redis_channel_monitor.py .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash monitor
RUN chown -R monitor:monitor /app
USER monitor

# Expose Prometheus metrics port
EXPOSE 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8082/metrics || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REDIS_URL=redis://redis:6379
ENV PROMETHEUS_PORT=8082

# Run the monitor
CMD ["python", "redis_channel_monitor.py"]