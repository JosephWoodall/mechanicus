import json
import numpy
import time
import logging
import argparse
import random
import hashlib
import yaml
import os
from datetime import datetime
from typing import Dict, Optional

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ml_training_dataset_generator.py")


class MLTrainingDatasetGenerator:
    """Generate training datasets for Mechanicus ML models.

    Creates datasets with EEG data and servo position mappings for training neural networks.
    """

    def __init__(
        self,
        n_channels: int = 5,
        n_servos: int = 3,
        total_positions: int = 100,
        servo_origin: list = None,
        servo_ceiling: list = None,
        hash_precision: int = 6,
        hash_step_size: int = 5,
        eeg_mean: float = 0.0,
        eeg_std: float = 1.0,
        training_samples: int = 10000,
        inference_samples_per_position: int = 1,
        config_file: str = None,
        output_file: str = None,
    ):
        """Initialize ML training dataset generator.

        Args:
            n_channels (int): Number of EEG channels to simulate
            n_servos (int): Number of servo motors
            total_positions (int): Total number of discrete positions
            servo_origin (list): Servo angle origins
            servo_ceiling (list): Servo angle ceilings
            hash_precision (int): Precision for position hashing
            hash_step_size (int): Step size for hash lookup
            eeg_mean (float): Baseline EEG mean
            eeg_std (float): Baseline EEG standard deviation
            training_samples (int): Number of training samples to generate
            inference_samples_per_position (int): Samples per position for inference
            config_file (str): Path to YAML configuration file
            output_file (str): Output JSON file path
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

                hash_precision = config.get('hash_lookup', {}).get(
                    'precision', hash_precision)
                hash_step_size = config.get('hash_lookup', {}).get(
                    'step_size', hash_step_size)

                n_channels = config.get('dataset', {}).get(
                    'n_eeg_channels', n_channels)
                eeg_mean = config.get('dataset', {}).get('eeg_mean', eeg_mean)
                eeg_std = config.get('dataset', {}).get('eeg_std', eeg_std)
                training_samples = config.get('dataset', {}).get(
                    'training_samples', training_samples)
                inference_samples_per_position = config.get('dataset', {}).get(
                    'inference_samples_per_position', inference_samples_per_position)

                if not output_file:
                    output_file = config.get('hash_lookup', {}).get(
                        'output_file', 'training_data.json')

                logger.info(f"Loaded configuration from: {config_file}")

        self.n_channels = n_channels
        self.n_servos = n_servos
        self.total_positions = total_positions
        self.servo_origin = numpy.array(servo_origin or [0, 0, 0])
        self.servo_ceiling = numpy.array(servo_ceiling or [180, 180, 180])

        self.hash_precision = hash_precision
        self.hash_step_size = hash_step_size

        self.baseline_mean = eeg_mean
        self.baseline_std = eeg_std
        self.training_samples = training_samples
        self.inference_samples_per_position = inference_samples_per_position

        self.output_file = output_file or "training_data.json"
        self.config = config if config_file else None

        self._generate_servo_positions()

        logger.info(f"MLTrainingDatasetGenerator Initialized:")
        logger.info(f"  EEG Channels: {self.n_channels}")
        logger.info(f"  Servos: {self.n_servos}")
        logger.info(f"  Total Positions: {len(self.servo_combinations)}")
        logger.info(f"  Training Samples: {self.training_samples}")
        logger.info(
            f"  EEG Mean/Std: {self.baseline_mean:.2f}/{self.baseline_std:.2f}")
        logger.info(f"  Output File: {self.output_file}")

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
                    os.path.join(os.path.dirname(__file__), 'shared',
                                 'config', config_file), 
                    os.path.join(os.path.dirname(__file__), 'src', 'shared',
                                 'config', config_file),   
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

            x = normalized_radius * numpy.sin(polar_angle) * numpy.cos(azimuth_angle)
            y = normalized_radius * numpy.sin(polar_angle) * numpy.sin(azimuth_angle)
            z = normalized_radius * numpy.cos(polar_angle)

            return numpy.array([x, y, z])
        else:
            position = numpy.zeros(3)
            for i, angle in enumerate(servo_angles[:3]):
                position[i] = angle / 180.0
            return position

    def _position_to_hash(self, position, precision=None):
        """Convert a 3D position to a hash value.

        Args:
            position (numpy.ndarray): 3D position [x, y, z].
            precision (int, optional): number of decimal places. Uses instance default if None.

        Returns:
            str: Hash value as a string.
        """
        if precision is None:
            precision = self.hash_precision

        rounded_position = numpy.round(position, precision)
        position_str = f"{rounded_position[0]:.{precision}f}_{rounded_position[1]:.{precision}f}_{rounded_position[2]:.{precision}f}"
        hash_value = hashlib.md5(position_str.encode()).hexdigest()[:12]
        return hash_value

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
        """Generate EEG data with anomaly spike.

        Returns:
            numpy.ndarray: numpy array of shape (n_channels,) with anomalous EEG values
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

    def generate_training_sample(self, sample_id: int, is_anomaly: bool = False) -> Dict:
        """Generate a single training sample with servo position data.

        Args:
            sample_id (int): Unique sample identifier
            is_anomaly (bool): Whether to generate anomalous EEG data

        Returns:
            Dict: Training sample dictionary with servo angles, position, and hash.
        """
        if is_anomaly:
            eeg_data = self.generate_anomaly_eeg()
        else:
            eeg_data = self.generate_baseline_eeg()

        index = numpy.random.randint(0, len(self.servo_combinations))
        servo_angles = self.servo_combinations[index]
        position = self.positions[index]
        position_hash = self._position_to_hash(position)

        sample = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": f"training_sample_{sample_id:06d}",
            "eeg_data": eeg_data.tolist(),
            "n_channels": self.n_channels,
            "is_anomaly": bool(is_anomaly),
            "servo_angles": servo_angles.tolist(),
            "position": position.tolist(),
            "position_hash": position_hash,
            "source": "MLTrainingDatasetGenerator",
            "metadata": {
                "baseline_mean": self.baseline_mean,
                "baseline_std": self.baseline_std,
                "sample_id": sample_id,
                "n_servos": self.n_servos,
                "servo_origin": self.servo_origin.tolist(),
                "servo_ceiling": self.servo_ceiling.tolist(),
                "hash_precision": self.hash_precision,
                "position_index": int(index),
            }
        }

        return sample

    def generate_hash_lookup(self) -> Dict:
        """Generate hash lookup table (same format as data_collector.py).

        Returns:
            Dict: Complete hash lookup structure
        """
        hash_lookup = {}
        reverse_lookup = {}

        for i, (servo_angles, position) in enumerate(zip(self.servo_combinations, self.positions)):
            position_hash = self._position_to_hash(position)

            hash_lookup[position_hash] = {
                "index": i,
                "servo_angles": servo_angles.tolist(),
                "position": position.tolist(),
                "servo_origin": self.servo_origin.tolist(),
                "servo_ceiling": self.servo_ceiling.tolist(),
                "n_servos": self.n_servos
            }

            position_key = f"{position[0]:.{self.hash_precision}f}_{position[1]:.{self.hash_precision}f}_{position[2]:.{self.hash_precision}f}"
            reverse_lookup[position_key] = position_hash

        lookup_data = {
            "metadata": {
                "total_positions": len(self.servo_combinations),
                "n_servos": self.n_servos,
                "servo_origin": self.servo_origin.tolist(),
                "servo_ceiling": self.servo_ceiling.tolist(),
                "hash_precision": self.hash_precision,
                "hash_step_size": self.hash_step_size,
                "generated_timestamp": datetime.now().isoformat(),
                "source": "MLTrainingDatasetGenerator",
                "config_file": self.config.get('hash_lookup', {}).get('output_file', 'training_data.json') if self.config else None
            },
            "hash_to_servo": hash_lookup,
            "position_to_hash": reverse_lookup
        }

        return lookup_data

    def generate_complete_dataset(self, anomaly_rate: float = 0.05) -> Dict:
        """Generate complete training dataset.

        Args:
            anomaly_rate (float): Proportion of samples that should be anomalies

        Returns:
            Dict: Complete dataset with samples and hash lookup
        """
        logger.info(
            f"Generating {self.training_samples} training samples with {anomaly_rate*100:.1f}% anomaly rate...")

        samples = []
        anomaly_count = 0

        for i in range(self.training_samples):
            is_anomaly = random.random() < anomaly_rate
            if is_anomaly:
                anomaly_count += 1

            sample = self.generate_training_sample(i + 1, is_anomaly)
            samples.append(sample)

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Generated {i + 1}/{self.training_samples} samples...")

        hash_lookup = self.generate_hash_lookup()

        dataset = {
            "dataset_metadata": {
                "total_samples": len(samples),
                "anomaly_samples": anomaly_count,
                "normal_samples": len(samples) - anomaly_count,
                "anomaly_rate": anomaly_count / len(samples) if len(samples) > 0 else 0,
                "n_channels": self.n_channels,
                "n_servos": self.n_servos,
                "total_positions": self.total_positions,
                "training_samples": self.training_samples,
                "inference_samples_per_position": self.inference_samples_per_position,
                "generated_timestamp": datetime.now().isoformat(),
                "source": "MLTrainingDatasetGenerator",
                "config_used": bool(self.config)
            },
            "training_samples": samples,
            "servo_hash_lookup": hash_lookup
        }

        logger.info(f"Dataset generation completed:")
        logger.info(f"  Total samples: {len(samples)}")
        logger.info(f"  Normal samples: {len(samples) - anomaly_count}")
        logger.info(f"  Anomaly samples: {anomaly_count}")
        logger.info(
            f"  Actual anomaly rate: {anomaly_count / len(samples) * 100:.2f}%")

        return dataset

    def save_dataset(self, dataset: Dict, output_file: str = None):
        """Save dataset to JSON file.

        Args:
            dataset (Dict): Complete dataset dictionary
            output_file (str, optional): Output file path. Uses instance default if None.
        """
        if output_file is None:
            output_file = self.output_file

        os.makedirs(os.path.dirname(output_file) if os.path.dirname(
            output_file) else '.', exist_ok=True)

        try:
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            logger.info(f"Dataset saved to: {output_file}")
            logger.info(
                f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to save dataset to {output_file}: {e}")
            raise


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="ML Training Dataset Generator for Mechanicus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--output",
        default="src/shared/data/training_data.json",
        help="Output JSON file path"
    )

    parser.add_argument(
        "--training-samples", type=int, default=1000,
        help="Number of training samples to generate"
    )

    parser.add_argument(
        "--anomaly-rate", type=float, default=0.05,
        help="Anomaly rate (0.0-1.0)"
    )

    parser.add_argument(
        "--channels", type=int, default=5,
        help="Number of EEG channels to simulate"
    )

    parser.add_argument(
        "--n-servos", type=int,
        help="Number of servo motors (overrides config)"
    )

    parser.add_argument(
        "--total-positions", type=int,
        help="Total number of discrete positions (overrides config)"
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

    generator_kwargs = {
        'n_channels': args.channels,
        'training_samples': args.training_samples,
        'config_file': args.config,
        'output_file': args.output,
    }

    if args.n_servos is not None:
        generator_kwargs['n_servos'] = args.n_servos
    if args.total_positions is not None:
        generator_kwargs['total_positions'] = args.total_positions
    if args.servo_origin is not None:
        generator_kwargs['servo_origin'] = args.servo_origin
    if args.servo_ceiling is not None:
        generator_kwargs['servo_ceiling'] = args.servo_ceiling

    try:
        generator = MLTrainingDatasetGenerator(**generator_kwargs)
        dataset = generator.generate_complete_dataset(
            anomaly_rate=args.anomaly_rate)
        generator.save_dataset(dataset)
        logger.info("Training dataset generation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Failed to generate training dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
