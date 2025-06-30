import redis
import json
import numpy
import time
import logging
import argparse
import random
import yaml
import os
from datetime import datetime
from typing import Dict, Optional

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eeg_data_publisher_service.py")


class EEGDataPublisherService:
    """Simulated EEG data publisher for testing Mechanicus services.

    Generates realistic EEG data with configurable anomaly spikes and publishes it to Redis channels for downstream service testing.
    """

    def __init__(
        self,
        redis_url: str = "redis://redis:6379",
        channel: str = "eeg_data",
        n_channels: int = 5,
        sampling_rate: float = 100.0,
        anomaly_rate: float = 0.05,
        n_servos: int = 3,
        total_positions: int = 100,
        servo_origin: list = [],
        servo_ceiling: list = [],
        eeg_mean: float = 0.0,
        eeg_std: float = 1.0,
        config_file: str = "",
    ):
        """Initialize simulated EEG data publisher service

        Args:
            redis_url (str, optional): Redis connection URL. Defaults to "redis://localhost:6379".
            channel (str, optional): Redis channel which to publish. Defaults to "eeg_data".
            n_channels (int, optional): number of eeg channels to simulate. Defaults to 5.
            sampling_rate (float, optional): sampling rate in Hz. Defaults to 100.0.
            anomaly_rate (float, optional): probability of generating anomaly spike (0-1). Defaults to 0.05.
            n_servos (int, optional): number of servo motors. Defaults to 3.
            total_positions (int, optional): total number of discrete positions. Defaults to 100.
            servo_origin (list, optional): servo angle origins. Defaults to [0, 0, 0].
            servo_ceiling (list, optional): servo angle ceilings. Defaults to [180, 180, 180].
            eeg_mean (float, optional): baseline EEG mean. Defaults to 0.0.
            eeg_std (float, optional): baseline EEG standard deviation. Defaults to 1.0.
            config_file (str, optional): Path to YAML configuration file.
        """

        if config_file:
            config = self._load_config(config_file)
            if config:
                n_servos = config.get('servo_config', {}).get(
                    'n_servos', n_servos)
                servo_origin = config.get('servo_config', {}).get(
                    'origin', servo_origin)
                servo_ceiling = config.get('servo_config', {}).get(
                    'ceiling', servo_ceiling)
                total_positions = config.get('servo_config', {}).get(
                    'total_positions', total_positions)

                n_channels = config.get('dataset', {}).get(
                    'n_eeg_channels', n_channels)
                eeg_mean = config.get('dataset', {}).get('eeg_mean', eeg_mean)
                eeg_std = config.get('dataset', {}).get('eeg_std', eeg_std)

                logger.info(f"Loaded configuration from: {config_file}")

        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.channel = channel
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.anomaly_rate = anomaly_rate
        self.sample_interval = 1.0 / self.sampling_rate

        self.n_servos = n_servos
        self.total_positions = total_positions
        self.servo_origin = numpy.array(servo_origin or [0, 0, 0])
        self.servo_ceiling = numpy.array(servo_ceiling or [180, 180, 180])

        self.baseline_mean = eeg_mean
        self.baseline_std = eeg_std
        self.anomaly_multiplier = 3.0

        self.total_samples = 0
        self.anomaly_count = 0
        self.start_time = time.time()

        self.config = config if config_file else None

        self._generate_servo_positions()

        logger.info(f"EEGDataPublisherService Initialized:")
        logger.info(f"  Redis URL: {redis_url}")
        logger.info(f"  Channel: {self.channel}")
        logger.info(f"  EEG Channels: {self.n_channels}")
        logger.info(f"  Sampling Rate: {self.sampling_rate} Hz")
        logger.info(f"  Anomaly Rate: {self.anomaly_rate * 100:.2f}%")
        logger.info(f"  Servos: {self.n_servos}")
        logger.info(f"  Total Positions: {len(self.servo_combinations)}")
        logger.info(
            f"  EEG Mean/Std: {self.baseline_mean:.2f}/{self.baseline_std:.2f}")

    def _load_config(self, config_file: str) -> Optional[Dict]:
        """Load configuration from YAML file.

        Args:
            config_file (str): Path to YAML configuration file

        Returns:
            Optional[Dict]: Configuration dictionary or None if failed
        """
        try:
            if not os.path.isabs(config_file):
                possible_paths = [
                    config_file,
                    os.path.join(os.path.dirname(__file__), config_file),
                    os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                 'shared', 'config', config_file),
                    os.path.join('/app', 'shared', 'config',
                                 config_file),
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        config_file = path
                        break
                else:
                    logger.error(
                        f"Configuration file not found in any of: {possible_paths}")
                    return None

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(
                    f"Successfully loaded configuration from: {config_file}")
                return config

        except Exception as e:
            logger.error(
                f"Failed to load configuration file {config_file}: {e}")
            return None

    def _generate_servo_positions(self):
        """Generate discrete servo angle combinations and their 3D positions."""
        steps_per_servo = max(
            3, int(numpy.ceil(self.total_positions ** (1 / self.n_servos))))

        servo_angles = []
        for i in range(self.n_servos):
            angles = numpy.linspace(
                self.servo_origin[i], self.servo_ceiling[i], steps_per_servo)
            servo_angles.append(angles)

        angle_grids = numpy.meshgrid(*servo_angles, indexing='ij')
        combinations = numpy.column_stack(
            [grid.flatten() for grid in angle_grids])

        if len(combinations) > self.total_positions:
            numpy.random.seed(1337)
            indices = numpy.random.choice(
                len(combinations), self.total_positions, replace=False)
            combinations = combinations[indices]
        elif len(combinations) < self.total_positions:
            additional_needed = self.total_positions - len(combinations)
            numpy.random.seed(1337)
            random_variations = []

            for _ in range(additional_needed):
                random_combo = []
                for i in range(self.n_servos):
                    random_angle = numpy.random.uniform(
                        self.servo_origin[i], self.servo_ceiling[i])
                    random_combo.append(random_angle)
                random_variations.append(random_combo)

            combinations = numpy.vstack(
                [combinations, numpy.array(random_variations)])

        self.servo_combinations = combinations

        self.positions = numpy.array([
            self._angles_to_cartesian_position(angles) for angles in self.servo_combinations
        ])

    def _angles_to_cartesian_position(self, servo_angles):
        """Convert servo angles to a 3D cartesian position.

        Args:
            servo_angles (numpy.ndarray): array of servo angles.

        Returns:
            numpy.ndarray: 3D cartesian position [x, y, z].
        """
        if len(servo_angles) >= 3:
            azimuth_angle = numpy.radians(servo_angles[0])
            polar_angle = numpy.radians(servo_angles[1])
            normalized_radius = servo_angles[2] / 180.0

            x = normalized_radius * \
                numpy.sin(polar_angle) * numpy.cos(azimuth_angle)
            y = normalized_radius * \
                numpy.sin(polar_angle) * numpy.sin(azimuth_angle)
            z = normalized_radius * numpy.cos(polar_angle)

            return numpy.array([x, y, z])
        else:
            position = numpy.zeros(3)
            for i, angle in enumerate(servo_angles[:3]):
                position[i] = angle / 180.0
            return position

    def _get_random_servo_position(self):
        """Get a random servo position from the generated combinations.

        Returns:
            tuple: (servo_angles, position)
        """
        index = numpy.random.randint(0, len(self.servo_combinations))
        servo_angles = self.servo_combinations[index]
        position = self.positions[index]

        return servo_angles, position

    def generate_baseline_eeg(self) -> numpy.ndarray:
        """Generate baseline EEG data (normal brain activity).

        Returns:
            numpy.ndarray: numpy array of shape (n_channels,) with baseline EEG values.
        """
        eeg_data = numpy.random.normal(
            self.baseline_mean, self.baseline_std, self.n_channels
        )
        time_factor = time.time() * 2 * numpy.pi
        for i in range(self.n_channels):
            frequency = 8 + (i * 3)
            amplitude = 0.3
            eeg_data[i] += amplitude * numpy.sin(frequency * time_factor)

        return eeg_data

    def generate_anomaly_eeg(self) -> numpy.ndarray:
        """
        Generate EEG data with anomaly spike.

        Returns:
            numpy array of shape (n_channels,) with anomalous EEG values
        """
        baseline = self.generate_baseline_eeg()

        spike_channels = random.sample(
            range(self.n_channels), random.randint(
                1, max(1, self.n_channels // 2))
        )

        for channel in spike_channels:
            spike_magnitude = random.uniform(2.5, 5.0)
            spike_direction = random.choice(
                [-1, 1])
            baseline[channel] += spike_direction * \
                spike_magnitude * self.baseline_std

        return baseline

    def generate_eeg_sample(self) -> Dict:
        """Generate a single EEG sample with servo position data.

        Returns:
            Dict: EEG sample dictionary with servo angles, position.
        """
        is_anomaly = random.random() < self.anomaly_rate

        if is_anomaly:
            eeg_data = self.generate_anomaly_eeg()
            self.anomaly_count += 1
        else:
            eeg_data = self.generate_baseline_eeg()

        servo_angles, position = self._get_random_servo_position()

        self.total_samples += 1

        sample = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": f"eeg_sample_{self.total_samples:06d}",
            "eeg_data": eeg_data.tolist(),
            "n_channels": self.n_channels,
            "is_anomaly": bool(is_anomaly),
            "servo_angles": servo_angles.tolist(),
            "position": position.tolist(),
            "source": "EEGDataPublisherService",
            "metadata": {
                "baseline_mean": self.baseline_mean,
                "baseline_std": self.baseline_std,
                "total_samples": self.total_samples,
                "anomaly_count": self.anomaly_count,
                "anomaly_rate": self.anomaly_count / self.total_samples if self.total_samples > 0 else 0,
                "n_servos": self.n_servos,
                "servo_origin": self.servo_origin.tolist(),
                "servo_ceiling": self.servo_ceiling.tolist(),
            }
        }

        return sample

    def publish_sample(self, sample: Dict) -> bool:
        """Publish EEG sample to Redis channel.

        Args:
            sample (Dict): EEG sample dictionary.

        Returns:
            bool: True if published successfully, False otherwise.
        """
        try:
            message = json.dumps(sample)
            self.redis_client.publish(self.channel, message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish sample: {e}")
            return False

    def clear_redis_data(self):
        """Clear all data from Redis server."""
        try:
            self.redis_client.flushall()
            #logger.info("Cleared all Redis data.")
        except Exception as e:
            logger.error(f"Failed to clear Redis data: {e}")

    def print_statistics(self):
        """Print current streaming statistics."""
        runtime = time.time() - self.start_time
        actual_rate = self.total_samples / runtime if runtime > 0 else 0
        actual_anomaly_rate = (
            self.anomaly_count / self.total_samples if self.total_samples > 0 else 0
        )

        logger.info(f"Streaming Statistics:")
        logger.info(f"  - Runtime: {runtime:.1f}s")
        logger.info(f"  - Generated Total Samples: {self.total_samples}")
        logger.info(f"  - Redis Total Keys: {self.redis_client.dbsize()}")
        logger.info(f"  - Generated Total Anomalies: {self.anomaly_count}")
        logger.info(f"  - Target Rate: {self.sampling_rate:.1f} Hz")
        logger.info(f"  - Actual Rate: {actual_rate:.1f} Hz")
        logger.info(f"  - Target Anomaly Rate: {self.anomaly_rate * 100:.1f}%")
        logger.info(
            f"  - Actual Anomaly Rate: {actual_anomaly_rate * 100:.1f}%")

        self.clear_redis_data()

    def run_continuously(self, duration: Optional[float] = None, verbose: bool = True):
        """Run the EEG data publisher continuously, generating and publishing samples at the configured sampling rate.

        Args:
            duration (Optional[float], optional): duration to run in seconds (none = indefinite). Defaults to None.
            verbose (bool, optional): whether to print periodic statistics. Defaults to True.
        """
        logger.info(
            f"Starting EEG data streaming to channel '{self.channel}'.")
        logger.info(
            f"Streaming at {self.sampling_rate} Hz with {self.anomaly_rate * 100:.1f}% anomaly rate."
        )

        if duration:
            logger.info(f"Running for {duration:.1f} seconds.")
        else:
            logger.info("Running indefinitely until interrupted.")

        start_time = time.time()
        last_stats_time = start_time

        try:
            while True:
                loop_start = time.time()

                sample = self.generate_eeg_sample()
                success = self.publish_sample(sample)

                if not success:
                    logger.warning("Failed to publish sample, continuing...")

                if verbose and (time.time() - last_stats_time) >= 10.0:
                    self.print_statistics()
                    last_stats_time = time.time()

                if duration and (time.time() - start_time) >= duration:
                    logger.info(
                        f"Completed {duration:.1f} seconds of streaming.")
                    break

                elapsed = time.time() - loop_start
                sleep_time = max(0, self.sample_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user.")

        logger.info("EEG data streaming completed.")

    def run_batch(self, n_samples: int, batch_interval: float = 1.0):
        """Run batch mode - send N samples then wait.

        Args:
            n_samples (int): number of samples per batch.
            batch_interval (float, optional): interval between batches in seconds. Defaults to 1.0.
        """

        logger.info(
            f"Starting batch mode: {n_samples} samples every {batch_interval:.1f} seconds."
        )

        try:
            while True:
                batch_start = time.time()

                logger.info(f"Sending batch of {n_samples} samples...")

                for i in range(n_samples):
                    sample = self.generate_eeg_sample()
                    self.publish_sample(sample)

                batch_time = time.time() - batch_start
                logger.info(f"batch sent in {batch_time:.2f} seconds.")

                time.sleep(batch_interval)
        except KeyboardInterrupt:
            logger.info("Batch mode interrupted by user.")

        self.print_statistics()


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="EEG Data Publisher Service for Mechanicus Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--redis-url", default="redis://redis:6379", help="Redis connection URL"
    )

    parser.add_argument(
        "--channel", default="eeg_data", help="Redis channel to publish to"
    )

    parser.add_argument(
        "--channels", type=int, default=5, help="Number of EEG channels to simulate"
    )

    parser.add_argument("--rate", type=float, default=100.0,
                        help="Sampling rate in Hz")

    parser.add_argument(
        "--anomaly-rate", type=float, default=0.05, help="Anomaly rate (0.0-1.0)"
    )

    parser.add_argument(
        "--duration",
        type=float,
        help="Duration to run in seconds (default: indefinite)",
    )

    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Run in batch mode instead of continuous",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples per batch (batch mode only)",
    )

    parser.add_argument(
        "--batch-interval",
        type=float,
        default=1.0,
        help="Interval between batches in seconds (batch mode only)",
    )

    parser.add_argument(
        "--quiet", action="store_true", help="Suppress periodic statistics output"
    )

    parser.add_argument(
        "--n-servos", type=int, help="Number of servo motors (overrides config)"
    )

    parser.add_argument(
        "--total-positions", type=int, help="Total number of discrete positions (overrides config)"
    )

    parser.add_argument(
        "--servo-origin", type=float, nargs='+',
        help="Servo angle origins (overrides config)"
    )

    parser.add_argument(
        "--servo-ceiling", type=float, nargs='+',
        help="Servo angle ceilings (overrides config)"
    )

    args = parser.parse_args()

    if not (0.0 <= args.anomaly_rate <= 1.0):
        parser.error("Anomaly rate must be between 0.0 and 1.0")

    if args.rate <= 0.0:
        parser.error("Sampling rate must be positive")

    publisher_kwargs = {
        'redis_url': args.redis_url,
        'channel': args.channel,
        'n_channels': args.channels,
        'sampling_rate': args.rate,
        'anomaly_rate': args.anomaly_rate,
        'config_file': args.config,
    }

    if args.n_servos is not None:
        publisher_kwargs['n_servos'] = args.n_servos
    if args.total_positions is not None:
        publisher_kwargs['total_positions'] = args.total_positions
    if args.servo_origin is not None:
        publisher_kwargs['servo_origin'] = args.servo_origin
    if args.servo_ceiling is not None:
        publisher_kwargs['servo_ceiling'] = args.servo_ceiling

    publisher = EEGDataPublisherService(**publisher_kwargs)

    try:
        publisher.redis_client.ping()
        logger.info("Connected to Redis server successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis server: {e}")
        return 1

    if args.batch_mode:
        publisher.run_batch(args.batch_size, args.batch_interval)
    else:
        publisher.run_continuously(args.duration, verbose=not args.quiet)

    return 0


if __name__ == "__main__":
    exit(main())
