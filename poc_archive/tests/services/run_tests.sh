#!/bin/bash
# filepath: tests/services/run_tests.sh

set -e

echo "Setting up test environment..."

# Function to check if Redis is running
check_redis() {
    redis-cli ping > /dev/null 2>&1
}

# Function to start Redis
start_redis() {
    echo "Starting Redis server..."
    
    # Use system Redis
    if command -v redis-server > /dev/null 2>&1; then
        echo "Starting Redis with redis-server..."
        redis-server --daemonize yes --port 6379 --loglevel warning
        sleep 2
        
        if check_redis; then
            echo "Redis started successfully with redis-server"
            return 0
        fi
    fi
    
    echo "Failed to start Redis. Please start it manually:"
    echo "   redis-server"
    echo "   or install Redis: sudo apt-get install redis-server"
    return 1
}

# Function to activate virtual environment
activate_venv() {
    # Look for virtual environment in common locations
    local venv_paths=(
        "venv/bin/activate"
        ".venv/bin/activate"
        "env/bin/activate"
        ".env/bin/activate"
        "../venv/bin/activate"
        "../../venv/bin/activate"
    )
    
    for venv_path in "${venv_paths[@]}"; do
        if [ -f "$venv_path" ]; then
            echo "Activating virtual environment: $venv_path"
            source "$venv_path"
            return 0
        fi
    done
    
    # Check if we're already in a virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Already in virtual environment: $VIRTUAL_ENV"
        return 0
    fi
    
    echo "Warning: No virtual environment found. Running with system Python."
    echo "Consider creating a virtual environment:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    return 1
}

# Function to stop Redis (if we started it)
cleanup_redis() {
    if [ "$REDIS_STARTED_BY_SCRIPT" = "true" ]; then
        echo "Cleaning up Redis..."
        echo "Stopping Redis server..."
        redis-cli shutdown 2>/dev/null || true
    fi
}

# Set up cleanup trap
trap cleanup_redis EXIT

# Check if Redis is already running
if check_redis; then
    echo "Redis is already running"
    REDIS_STARTED_BY_SCRIPT="false"
else
    echo "Redis is not running, starting it..."
    if start_redis; then
        REDIS_STARTED_BY_SCRIPT="true"
    else
        exit 1
    fi
fi

# Verify Redis is accessible
echo "Verifying Redis connection..."
if ! check_redis; then
    echo "Redis connection failed after startup attempt"
    exit 1
fi

echo "Redis is ready for testing"

# Activate virtual environment
echo "Setting up Python environment..."
cd "$(dirname "$0")/../.." # Go to project root
activate_venv

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/services/test_message_broker.py -v

# Run integration test
echo "Running integration test..."

# Start EEG publisher in background
echo "Starting EEG publisher..."
python tests/services/eeg_data_publisher_service.py \
    --channel "integration_test" \
    --rate 50 \
    --duration 10 \
    --quiet &

PUBLISHER_PID=$!

# Wait a moment for publisher to start
sleep 2

# Run integration tests
echo "Running integration tests..."
python -c "
import redis
import json
import time
import sys

try:
    client = redis.from_url('redis://localhost:6379', decode_responses=True)
    pubsub = client.pubsub()
    pubsub.subscribe('integration_test')

    print('Listening for messages...')
    messages = []
    timeout = time.time() + 8  # Give more time

    while time.time() < timeout and len(messages) < 3:
        message = pubsub.get_message(timeout=1.0)
        if message and message['type'] == 'message':
            data = json.loads(message['data'])
            messages.append(data)
            print(f'Received message {len(messages)}: sample_id={data[\"sample_id\"]}')

    print(f'Received {len(messages)} messages')
    
    if len(messages) >= 1:
        print('Integration test passed!')
        sys.exit(0)
    else:
        print('Integration test failed: No messages received')
        sys.exit(1)
        
except Exception as e:
    print(f'Integration test failed: {e}')
    sys.exit(1)
"

INTEGRATION_EXIT_CODE=$?

wait $PUBLISHER_PID 2>/dev/null || true

if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo "All tests completed successfully!"
else
    echo "Integration tests failed"
    exit 1
fi