import redis
import json
import logging
import os
import sys
import docker
import subprocess
from typing import Dict, Any, List
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from collections import deque
import numpy as np
import signal
import time
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MechanicusRedisChannelMonitor:
    """
    Redis channel monitor for Mechanicus project.
    Tracks total record counts per Redis channel and monitors model performance.
    """

    def __init__(self, redis_url: str = "redis://redis:6379", prometheus_port: int = 8082):
        self.redis_url = redis_url
        self.prometheus_port = prometheus_port
        self.running = False

        try:
            self.redis_client = redis.from_url(
                redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            sys.exit(1)

        self.channels_to_monitor = [
            'eeg_data',
            'eeg_anomalies',
            'eeg_data_processed',
            'predicted_servo_angles',
            'servo_execution_feedback'  # Added for real accuracy tracking
        ]

        # Model performance tracking
        self.model_performance_window = deque(
            maxlen=100)  # Last 100 predictions
        self.performance_threshold = float(
            os.getenv('MODEL_PERFORMANCE_THRESHOLD', '0.8'))
        self.min_samples_for_evaluation = int(
            os.getenv('MIN_SAMPLES_FOR_EVALUATION', '50'))
        self.retraining_cooldown_minutes = int(
            os.getenv('RETRAINING_COOLDOWN_MINUTES', '60'))
        self.last_retraining_time = None
        self.retraining_enabled = os.getenv(
            'RETRAINING_ENABLED', 'true').lower() == 'true'

        # Live validation data collection (replacing static file)
        self.validation_samples = deque(maxlen=200)  # Keep last 200 samples
        self.validation_interval_minutes = int(
            os.getenv('VALIDATION_INTERVAL_MINUTES', '30'))
        self.last_validation_time = None
        self.max_acceptable_error = float(
            os.getenv('MAX_ACCEPTABLE_ERROR_DEGREES', '30.0'))

        # Pending predictions tracking (for matching with feedback)
        self.pending_predictions = {}  # timestamp -> prediction_data
        self.prediction_timeout_seconds = int(
            os.getenv('PREDICTION_TIMEOUT_SECONDS', '10'))

        # Docker client for triggering retraining
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None

        # Initialize Prometheus metrics
        self._setup_prometheus_metrics()
        self._setup_retraining_metrics()

        # Subscribe to channels
        self._subscribe_to_channels()

        # Start Prometheus server
        self._start_prometheus_server()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics for record counting."""
        self.channel_records_total = Counter(
            'mechanicus_redis_channel_records_total',
            'Total records published to each Redis channel',
            ['channel']
        )

        self.total_records = Counter(
            'mechanicus_redis_total_records',
            'Total records published across all Redis channels'
        )

        self.active_channels = Gauge(
            'mechanicus_redis_active_channels',
            'Number of Redis channels being monitored'
        )

        for channel in self.channels_to_monitor:
            self.channel_records_total.labels(channel=channel)._value.set(0)

        logger.info("Prometheus metrics initialized - tracking record counts")

    def _setup_retraining_metrics(self):
        """Setup metrics for model retraining monitoring."""
        # Model performance metrics
        self.model_accuracy = Gauge(
            'mechanicus_model_accuracy',
            'Current model accuracy based on recent predictions'
        )

        self.model_confidence = Histogram(
            'mechanicus_model_confidence',
            'Model prediction confidence scores'
        )

        self.retraining_triggers = Counter(
            'mechanicus_retraining_triggers_total',
            'Total number of retraining triggers'
        )

        self.retraining_status = Gauge(
            'mechanicus_retraining_status',
            'Current retraining status (0=idle, 1=running, 2=failed)'
        )

        self.last_retraining_timestamp = Gauge(
            'mechanicus_last_retraining_timestamp',
            'Timestamp of last retraining attempt'
        )

        # Validation metrics
        self.validation_accuracy = Gauge(
            'mechanicus_model_validation_accuracy',
            'Model accuracy on validation dataset'
        )

        self.validation_samples_count = Gauge(
            'mechanicus_validation_samples_count',
            'Number of samples in validation buffer'
        )

        self.servo_execution_success_rate = Gauge(
            'mechanicus_servo_execution_success_rate',
            'Rate of successful servo executions'
        )

        self.prediction_accuracy_histogram = Histogram(
            'mechanicus_prediction_accuracy',
            'Distribution of prediction accuracy scores'
        )

        # Retraining enabled flag
        self.retraining_enabled_metric = Gauge(
            'mechanicus_retraining_enabled',
            'Whether automatic retraining is enabled (1=enabled, 0=disabled)'
        )
        self.retraining_enabled_metric.set(1 if self.retraining_enabled else 0)

        logger.info("Retraining metrics initialized")

    def _subscribe_to_channels(self):
        """Subscribe to all Redis channels with retry logic."""
        max_retries = 5
        retry_delay = 2

        for channel in self.channels_to_monitor:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.pubsub.subscribe(channel)
                    logger.info(
                        f"Successfully subscribed to Redis channel: {channel}")
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"Failed to subscribe to channel {channel} (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to subscribe to channel {channel} after {max_retries} attempts")

        self.active_channels.set(len(self.channels_to_monitor))

    def _start_prometheus_server(self):
        """Start Prometheus metrics HTTP server."""
        try:
            start_http_server(self.prometheus_port)
            logger.info(
                f"Prometheus metrics server started on port {self.prometheus_port}")
            logger.info(
                f"Metrics available at: http://localhost:{self.prometheus_port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            sys.exit(1)

    def _collect_validation_sample(self, eeg_sample: Dict[str, Any]) -> bool:
        """Collect EEG samples for validation purposes."""
        try:
            # Extract validation data from live EEG stream
            validation_sample = {
                "timestamp": eeg_sample.get("timestamp", datetime.now().isoformat()),
                "eeg_data": eeg_sample.get("eeg_data", []),
                "expected_servo_angles": eeg_sample.get("servo_angles", []),
                "expected_position": eeg_sample.get("position", []),
                "is_anomaly": eeg_sample.get("is_anomaly", False),
                "sample_id": eeg_sample.get("sample_id", f"sample_{int(time.time() * 1000)}")
            }

            # Only collect samples that have both EEG data and expected servo angles
            if validation_sample["eeg_data"] and validation_sample["expected_servo_angles"]:
                self.validation_samples.append(validation_sample)
                self.validation_samples_count.set(len(self.validation_samples))
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to collect validation sample: {e}")
            return False

    def _process_channel_message(self, channel: str, data: Dict[str, Any]):
        """Process message from any channel - count records and track performance."""
        try:
            # Increment record counters
            self.channel_records_total.labels(channel=channel).inc()
            self.total_records.inc()

            # Collect validation samples from EEG data
            if channel == 'eeg_data':
                self._collect_validation_sample(data)

            # Track model performance for predicted_servo_angles
            elif channel == 'predicted_servo_angles':
                self._track_prediction(data)

            # Track servo execution feedback for real accuracy
            elif channel == 'servo_execution_feedback':
                self._track_servo_execution_feedback(data)

            # Run validation periodically
            if self._should_run_validation():
                self._run_model_validation()

            # Check if retraining is needed
            if self._should_trigger_retraining():
                self._trigger_retraining()

            # Clean up old pending predictions
            self._cleanup_old_predictions()

            # Logging
            current_count = self.channel_records_total.labels(
                channel=channel)._value._value
            total_count = self.total_records._value._value
            #logger.info(
            #    f"Channel '{channel}' record count: {current_count}, Total: {total_count}")

            logger.debug(
                f"Received data on channel '{channel}': {json.dumps(data)[:100]}...")

        except Exception as e:
            logger.error(f"Error processing message from {channel}: {e}")

    def _track_prediction(self, prediction_data: Dict[str, Any]):
        """Track prediction for later comparison with execution feedback."""
        try:
            timestamp = prediction_data.get(
                'timestamp', datetime.now().isoformat())

            # Store prediction for later matching with feedback
            self.pending_predictions[timestamp] = {
                'prediction_data': prediction_data,
                'created_at': time.time()
            }

            # Extract and track confidence if available
            confidence = prediction_data.get('confidence', 1.0)
            self.model_confidence.observe(confidence)

            logger.debug(f"Tracked prediction at {timestamp}")

        except Exception as e:
            logger.error(f"Error tracking prediction: {e}")

    def _track_servo_execution_feedback(self, feedback_data: Dict[str, Any]):
        """Track servo execution feedback for real accuracy calculation."""
        try:
            # Extract feedback information
            predicted_angles = feedback_data.get('predicted_angles', [])
            actual_angles = feedback_data.get('actual_angles', [])
            success = feedback_data.get('success', False)
            execution_timestamp = feedback_data.get('timestamp')
            prediction_timestamp = feedback_data.get('prediction_timestamp')

            # Calculate real accuracy if we have both predicted and actual angles
            if predicted_angles and actual_angles:
                accuracy = self._calculate_real_accuracy(
                    predicted_angles, actual_angles)

                # Add to performance window
                self.model_performance_window.append(accuracy)

                # Update metrics
                current_accuracy = np.mean(self.model_performance_window)
                self.model_accuracy.set(current_accuracy)
                self.prediction_accuracy_histogram.observe(accuracy)

                logger.info(
                    f"Real accuracy calculated: {accuracy:.3f}, Current avg: {current_accuracy:.3f}")

            # Track execution success rate
            if success is not None:
                # Update success rate (this is a simplified version)
                # In practice, you'd want to track this over a window
                self.servo_execution_success_rate.set(1.0 if success else 0.0)

            # Remove from pending predictions if matched
            if prediction_timestamp and prediction_timestamp in self.pending_predictions:
                del self.pending_predictions[prediction_timestamp]
                logger.debug(
                    f"Matched prediction feedback for {prediction_timestamp}")

        except Exception as e:
            logger.error(f"Error tracking servo execution feedback: {e}")

    def _calculate_real_accuracy(self, predicted_angles: List[float], actual_angles: List[float]) -> float:
        """Calculate real accuracy based on predicted vs actual servo angles."""
        try:
            if len(predicted_angles) != len(actual_angles):
                logger.warning(
                    f"Angle count mismatch: predicted={len(predicted_angles)}, actual={len(actual_angles)}")
                return 0.0

            # Calculate Mean Absolute Error
            errors = [abs(pred - act)
                      for pred, act in zip(predicted_angles, actual_angles)]
            mae = np.mean(errors)

            # Convert to accuracy score (0-1)
            accuracy = max(0.0, 1.0 - (mae / self.max_acceptable_error))

            logger.debug(
                f"Calculated accuracy: MAE={mae:.2f}°, Accuracy={accuracy:.3f}")
            return accuracy

        except Exception as e:
            logger.error(f"Error calculating real accuracy: {e}")
            return 0.0

    def _should_run_validation(self) -> bool:
        """Check if we should run validation against live data."""
        if not self.last_validation_time:
            return len(self.validation_samples) >= self.min_samples_for_evaluation

        time_since_last = datetime.now() - self.last_validation_time
        return time_since_last >= timedelta(minutes=self.validation_interval_minutes)

    def _run_model_validation(self):
        """Run model validation using collected live data."""
        try:
            logger.info("Running model validation...")

            if len(self.validation_samples) < self.min_samples_for_evaluation:
                logger.warning(
                    f"Not enough validation samples: {len(self.validation_samples)}/{self.min_samples_for_evaluation}")
                return

            validation_errors = []
            prediction_times = []

            # Use a subset of validation samples for testing
            samples_to_test = list(
                self.validation_samples)[-self.min_samples_for_evaluation:]

            for sample in samples_to_test:
                start_time = time.time()

                # Get model prediction using live data
                predicted_angles = self._get_model_prediction(
                    sample["eeg_data"])

                prediction_time = time.time() - start_time
                prediction_times.append(prediction_time)

                if predicted_angles is not None and sample["expected_servo_angles"]:
                    # Calculate error against expected servo angles
                    expected_angles = np.array(sample["expected_servo_angles"])
                    predicted_angles = np.array(predicted_angles)

                    if len(expected_angles) == len(predicted_angles):
                        error = np.mean(
                            np.abs(predicted_angles - expected_angles))
                        validation_errors.append(error)

                        # Check if prediction timeout exceeded
                        if prediction_time > self.prediction_timeout_seconds:
                            logger.warning(
                                f"Prediction timeout exceeded: {prediction_time:.2f}s > {self.prediction_timeout_seconds}s")
                    else:
                        logger.warning(
                            f"Angle dimension mismatch: expected {len(expected_angles)}, got {len(predicted_angles)}")

            if validation_errors:
                avg_error = np.mean(validation_errors)
                avg_prediction_time = np.mean(prediction_times)

                # Convert error to accuracy
                accuracy = max(
                    0.0, 1.0 - (avg_error / self.max_acceptable_error))

                logger.info(f"Validation Results:")
                logger.info(f"  - Average Error: {avg_error:.2f} degrees")
                logger.info(f"  - Average Accuracy: {accuracy:.3f}")
                logger.info(
                    f"  - Average Prediction Time: {avg_prediction_time:.3f}s")
                logger.info(f"  - Samples Validated: {len(validation_errors)}")

                # Update metrics
                self.validation_accuracy.set(accuracy)

                # Add validation results to performance window
                self.model_performance_window.append(accuracy)

                # Update current model accuracy
                current_accuracy = np.mean(self.model_performance_window)
                self.model_accuracy.set(current_accuracy)

                # Check if retraining is needed
                if current_accuracy < self.performance_threshold:
                    logger.warning(
                        f"Model performance degraded: {current_accuracy:.3f} < {self.performance_threshold}")

            else:
                logger.warning(
                    "No valid predictions obtained during validation")

            self.last_validation_time = datetime.now()

        except Exception as e:
            logger.error(f"Model validation failed: {e}")

    def _get_model_prediction(self, eeg_data: List[float]) -> List[float]:
        """Get prediction from inference model via Redis."""
        try:
            # Create prediction request
            prediction_request = {
                "timestamp": time.time(),
                "eeg_data": eeg_data,
                "request_id": f"validation_{int(time.time() * 1000)}"
            }

            # For now, return a mock prediction since we don't have the full inference pipeline
            # In production, you would:
            # 1. Publish to inference request channel
            # 2. Wait for response on inference response channel
            # 3. Handle timeouts appropriately

            # Mock prediction based on EEG data patterns
            if len(eeg_data) >= 3:
                # Simple mock: convert first 3 EEG values to servo angles
                mock_angles = [
                    max(0, min(180, abs(eeg_data[0]) * 100)),
                    max(0, min(180, abs(eeg_data[1]) * 100)),
                    max(0, min(180, abs(eeg_data[2]) * 100))
                ]
                return mock_angles

            return [45.0, 90.0, 135.0]  # Default mock angles

        except Exception as e:
            logger.error(f"Failed to get model prediction: {e}")
            return None

    def _cleanup_old_predictions(self):
        """Clean up old pending predictions that haven't received feedback."""
        try:
            current_time = time.time()
            expired_predictions = []

            for timestamp, prediction_info in self.pending_predictions.items():
                if current_time - prediction_info['created_at'] > self.prediction_timeout_seconds:
                    expired_predictions.append(timestamp)

            for timestamp in expired_predictions:
                del self.pending_predictions[timestamp]
                logger.debug(f"Cleaned up expired prediction: {timestamp}")

        except Exception as e:
            logger.error(f"Error cleaning up old predictions: {e}")

    def _should_trigger_retraining(self) -> bool:
        """Determine if retraining should be triggered."""
        try:
            # Check if retraining is enabled
            if not self.retraining_enabled:
                return False

            # Check if we have enough samples
            if len(self.model_performance_window) < self.min_samples_for_evaluation:
                return False

            # Check cooldown period
            if self.last_retraining_time:
                time_since_last = datetime.now() - self.last_retraining_time
                if time_since_last < timedelta(minutes=self.retraining_cooldown_minutes):
                    return False

            # Check if performance has degraded below threshold
            current_accuracy = np.mean(self.model_performance_window)
            if current_accuracy < self.performance_threshold:
                logger.warning(
                    f"Model performance below threshold: {current_accuracy:.3f} < {self.performance_threshold}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking retraining conditions: {e}")
            return False

    def _trigger_retraining(self):
        """Trigger the offline training pipeline."""
        try:
            logger.info("Triggering model retraining...")
            self.retraining_triggers.inc()
            self.retraining_status.set(1)  # Set to running
            self.last_retraining_time = datetime.now()
            self.last_retraining_timestamp.set(time.time())

            # Use Docker Compose to trigger retraining
            self._run_docker_compose_training()

        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")
            self.retraining_status.set(2)  # Set to failed

    def _run_docker_compose_training(self):
        """Run the offline training using Docker Compose."""
        try:
            logger.info("Starting Docker Compose training pipeline...")

            # Path to the training compose file
            compose_file = os.getenv(
                'TRAINING_COMPOSE_FILE', '/app/docker-compose.offline_training.yml')

            if not os.path.exists(compose_file):
                logger.error(
                    f"Training compose file not found: {compose_file}")
                self.retraining_status.set(2)
                return

            # Stop any existing training containers
            self._cleanup_existing_training_containers()

            # Run the training pipeline
            cmd = [
                "docker-compose",
                "-f", compose_file,
                "up",
                "--build",
                "--remove-orphans"
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            # Run training in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(
                    timeout=3600)  # 1 hour timeout

                if process.returncode == 0:
                    logger.info("Retraining completed successfully")
                    self.retraining_status.set(0)  # Set to idle
                    # Clear performance window to start fresh
                    self.model_performance_window.clear()
                else:
                    logger.error(
                        f"Retraining failed with return code {process.returncode}")
                    logger.error(f"stderr: {stderr}")
                    self.retraining_status.set(2)  # Set to failed

            except subprocess.TimeoutExpired:
                logger.error("Retraining timed out")
                process.kill()
                self.retraining_status.set(2)

        except Exception as e:
            logger.error(f"Error running Docker Compose training: {e}")
            self.retraining_status.set(2)

    def _cleanup_existing_training_containers(self):
        """Clean up any existing training containers."""
        try:
            if not self.docker_client:
                return

            # Stop containers from mechanicus-offline-training project
            existing_containers = self.docker_client.containers.list(
                filters={
                    "label": "com.docker.compose.project=mechanicus-offline-training"}
            )

            for container in existing_containers:
                logger.info(
                    f"Stopping existing training container: {container.name}")
                container.stop()
                container.remove()

        except Exception as e:
            logger.warning(f"Error cleaning up existing containers: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def start_monitoring(self):
        """Start monitoring Redis channels."""
        logger.info("Starting Mechanicus Redis channel monitoring...")
        logger.info(f"Monitoring channels: {self.channels_to_monitor}")
        logger.info(
            f"Model performance threshold: {self.performance_threshold}")
        logger.info(
            f"Minimum samples for evaluation: {self.min_samples_for_evaluation}")
        logger.info(
            f"Retraining cooldown: {self.retraining_cooldown_minutes} minutes")
        logger.info(
            f"Validation interval: {self.validation_interval_minutes} minutes")
        logger.info(
            f"Automatic retraining: {'ENABLED' if self.retraining_enabled else 'DISABLED'}")
        logger.info(
            f"Using live data for validation (buffer size: {self.validation_samples.maxlen})")

        self.running = True

        try:
            for message in self.pubsub.listen():
                if not self.running:
                    break

                if message['type'] == 'message':
                    channel = message['channel']

                    try:
                        data = json.loads(message['data'])
                        self._process_channel_message(channel, data)

                        # Log milestone counts
                        current_count = self.channel_records_total.labels(
                            channel=channel)._value._value
                        if current_count % 100 == 0:
                            logger.info(
                                f"Channel '{channel}' milestone: {current_count} records")

                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to decode JSON from channel {channel} - skipping record")
                    except Exception as e:
                        logger.error(
                            f"Error processing message from {channel}: {e}")

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up Redis channel monitor...")

        # Log final counts
        logger.info("Final record counts:")
        for channel in self.channels_to_monitor:
            try:
                count = self.channel_records_total.labels(
                    channel=channel)._value._value
                logger.info(f"  {channel}: {count} records")
            except:
                logger.info(f"  {channel}: 0 records")

        # Log final model performance
        if len(self.model_performance_window) > 0:
            final_accuracy = np.mean(self.model_performance_window)
            logger.info(f"Final model accuracy: {final_accuracy:.3f}")

        # Log validation buffer status
        logger.info(
            f"Validation samples collected: {len(self.validation_samples)}")

        # Log pending predictions
        if self.pending_predictions:
            logger.info(
                f"Pending predictions: {len(self.pending_predictions)}")

        # Close Redis connections
        try:
            if hasattr(self, 'pubsub'):
                self.pubsub.close()
            if hasattr(self, 'redis_client'):
                self.redis_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        # Close Docker client
        try:
            if self.docker_client:
                self.docker_client.close()
        except Exception as e:
            logger.error(f"Error closing Docker client: {e}")

        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
    prometheus_port = int(os.getenv('PROMETHEUS_PORT', '8082'))

    monitor = MechanicusRedisChannelMonitor(
        redis_url=redis_url,
        prometheus_port=prometheus_port
    )

    try:
        monitor.start_monitoring()
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
