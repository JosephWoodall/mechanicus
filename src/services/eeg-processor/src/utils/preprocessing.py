import redis
import json
import logging
import numpy
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGMonitorService:
    def __init__(self, redis_url="redis://localhost:6379",
                 input_channel="eeg_data",
                 output_channel="eeg_data_processed"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.pubsub = self.redis_client.pubsub()
        logger.info("=" * 50)
        logger.info(f"EEGMonitorService Initialized:")
        logger.info(f"    Input Channel: {self.input_channel}")
        logger.info(f"    Output Channel: {self.output_channel}")
        logger.info("=" * 50)

    def detect_anomaly(self, eeg_data):
        """Anomaly detection logic"""
        if not eeg_data or len(eeg_data) == 0:
            logger.warning("Received empty EEG data for anomaly detection.")
            return False

        if isinstance(eeg_data, list):
            eeg_data = numpy.array(eeg_data)

        threshold = 2.5
        return numpy.any(numpy.abs(eeg_data) > threshold)

    def publish_processed_anomaly(self, original_sample, detected_anomaly):
        """Publish processed EEG anomaly to output channel"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "original_sample": original_sample,
            "processor_anomaly_detected": bool(detected_anomaly),
            "source": "eeg-processor",
            "processing_timestamp": datetime.now().isoformat()
        }

        try:
            self.redis_client.publish(
                self.output_channel,
                json.dumps(message)
            )
        except Exception as e:
            logger.error(f"Failed to publish processed anomaly: {e}")

    def process_eeg_sample(self, sample_data):
        """Process incoming EEG sample and detect anomalies

        Args:
            sample_data (list): The EEG sample data to process.
        """

        try:
            eeg_data = sample_data.get('eeg_data', [])

            detected_anomaly = self.detect_anomaly(eeg_data)

            if detected_anomaly:
                self.publish_processed_anomaly(sample_data, detected_anomaly)
        except Exception as e:
            logger.error(f"Error processing EEG sample: {e}")
            self.publish_processed_anomaly(sample_data, False)

    def run(self):
        """Main monitoring loop - subscribe to input channel.
        """

        self.pubsub.subscribe(self.input_channel)
        logger.info(f"Subscribed to channel: {self.input_channel}")

        try:
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        sample_data = json.loads(message['data'])
                        self.process_eeg_sample(sample_data)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON: {e}")
                    except KeyboardInterrupt as e:
                        logger.info(
                            "KeyboardInterrupt received, shutting down...")
                        break

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.pubsub.close()
            logger.info("EEG Monitor Service stopped.")


if __name__ == "__main__":
    import os

    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    input_channel = os.getenv('INPUT_CHANNEL', 'eeg_data')
    output_channel = os.getenv('OUTPUT_CHANNEL', 'eeg_data_processed')

    service = EEGMonitorService(
        redis_url=redis_url,
        input_channel=input_channel,
        output_channel=output_channel
    )
    service.run()
