import redis
import json
import logging
import os
import sys
from typing import Dict, Any
from prometheus_client import Counter, Gauge, start_http_server
import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MechanicusRedisChannelMonitor:
    """
    Redis channel monitor for Mechanicus project.
    Tracks total record counts per Redis channel.
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
            'predicted_servo_angles'
        ]

        self._setup_prometheus_metrics()

        self._subscribe_to_channels()

        self._start_prometheus_server()

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

        logger.info(
            "Prometheus metrics initialized - tracking record counts only")

    def _subscribe_to_channels(self):
        """Subscribe to all Redis channels."""
        for channel in self.channels_to_monitor:
            try:
                self.pubsub.subscribe(channel)
                logger.info(f"Subscribed to Redis channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to subscribe to channel {channel}: {e}")

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

    def _process_channel_message(self, channel: str, data: Dict[str, Any]):
        """Process message from any channel - just count records."""
        try:
            self.channel_records_total.labels(channel=channel).inc()

            self.total_records.inc()

            current_count = self.channel_records_total.labels(
                channel=channel)._value._value
            total_count = self.total_records._value._value
            logger.info(
                f"Channel '{channel}' record count: {current_count}, Total: {total_count}")

            logger.info(
                f"Received data on channel '{channel}': {json.dumps(data)[:100]}...")

        except Exception as e:
            logger.error(f"Error processing message from {channel}: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def start_monitoring(self):
        """Start monitoring Redis channels."""
        logger.info("Starting Mechanicus Redis channel monitoring...")
        logger.info(f"Monitoring channels: {self.channels_to_monitor}")
        logger.info("Tracking: Total record counts per channel")

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

                        current_count = self.channel_records_total.labels(
                            channel=channel)._value._value
                        if current_count % 100 == 0:
                            logger.info(
                                f"Channel '{channel}' total records: {current_count}")

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

        logger.info("Final record counts:")
        for channel in self.channels_to_monitor:
            try:
                count = self.channel_records_total.labels(
                    channel=channel)._value._value
                logger.info(f"  {channel}: {count} records")
            except:
                logger.info(f"  {channel}: 0 records")

        try:
            if hasattr(self, 'pubsub'):
                self.pubsub.close()
            if hasattr(self, 'redis_client'):
                self.redis_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
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
