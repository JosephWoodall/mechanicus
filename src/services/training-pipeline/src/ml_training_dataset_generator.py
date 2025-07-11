import json
import numpy
import time
import logging
import argparse
import random
import yaml
import os
from datetime import datetime
from typing import Dict, List, Optional

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
        eeg_mean: float = 0.0,
        eeg_std: float = 1.0,
        training_samples: int = 1000,
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


                n_channels = config.get('dataset', {}).get(
                    'n_eeg_channels', n_channels)
                eeg_mean = config.get('dataset', {}).get('eeg_mean', eeg_mean)
                eeg_std = config.get('dataset', {}).get('eeg_std', eeg_std)
                training_samples = config.get('dataset', {}).get(
                    'training_samples', training_samples)
                inference_samples_per_position = config.get('dataset', {}).get(
                    'inference_samples_per_position', inference_samples_per_position)

                logger.info(f"Loaded configuration from: {config_file}")

        self.n_channels = n_channels
        self.n_servos = n_servos
        self.total_positions = total_positions
        self.servo_origin = numpy.array(servo_origin or [0, 0, 0])
        self.servo_ceiling = numpy.array(servo_ceiling or [180, 180, 180])



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
                    os.path.join('/app', 'shared', 'config',
                                 config_file),  
                    os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                 'shared', 'config', config_file),
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
            Dict: Training sample dictionary with servo angles, position.
        """
        if is_anomaly:
            eeg_data = self.generate_anomaly_eeg()
        else:
            eeg_data = self.generate_baseline_eeg()

        index = numpy.random.randint(0, len(self.servo_combinations))
        servo_angles = self.servo_combinations[index]
        position = self.positions[index]

        sample = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": f"training_sample_{sample_id:06d}",
            "eeg_data": eeg_data.tolist(),
            "n_channels": self.n_channels,
            "is_anomaly": bool(is_anomaly),
            "servo_angles": servo_angles.tolist(),
            "position": position.tolist(),
            "source": "MLTrainingDatasetGenerator",
            "metadata": {
                "baseline_mean": self.baseline_mean,
                "baseline_std": self.baseline_std,
                "sample_id": sample_id,
                "n_servos": self.n_servos,
                "servo_origin": self.servo_origin.tolist(),
                "servo_ceiling": self.servo_ceiling.tolist(),
                "position_index": int(index),
            }
        }

        return sample

    

    def generate_complete_dataset(self, anomaly_rate: float = 0.05) -> Dict:
        """Generate complete training dataset.

        Args:
            anomaly_rate (float): Proportion of samples that should be anomalies

        Returns:
            Dict: Complete dataset with samples
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
                "config_used": bool(self.config),
                "dataset_type": "training"
            },
            "training_samples": samples,
        }

        logger.info(f"Dataset generation completed:")
        logger.info(f"  Total samples: {len(samples)}")
        logger.info(f"  Normal samples: {len(samples) - anomaly_count}")
        logger.info(f"  Anomaly samples: {anomaly_count}")
        logger.info(
            f"  Actual anomaly rate: {anomaly_count / len(samples) * 100:.2f}%")

        return dataset

    def generate_inference_dataset(self, anomaly_rate: float = 0.0) -> Dict:
        """Generate inference dataset with one sample per position.

        Args:
            anomaly_rate (float): Proportion of samples that should be anomalies

        Returns:
            Dict: Complete inference dataset with samples 
        """
        logger.info(
            f"Generating inference dataset with {self.inference_samples_per_position} sample(s) per position...")

        samples = []
        anomaly_count = 0
        sample_counter = 0

        for position_idx in range(len(self.servo_combinations)):
            for sample_per_pos in range(self.inference_samples_per_position):
                sample_counter += 1

                is_anomaly = random.random() < anomaly_rate
                if is_anomaly:
                    anomaly_count += 1

                if is_anomaly:
                    eeg_data = self.generate_anomaly_eeg()
                else:
                    eeg_data = self.generate_baseline_eeg()

                servo_angles = self.servo_combinations[position_idx]
                position = self.positions[position_idx]

                sample = {
                    "timestamp": datetime.now().isoformat(),
                    "sample_id": f"inference_sample_{sample_counter:06d}",
                    "eeg_data": eeg_data.tolist(),
                    "n_channels": self.n_channels,
                    "is_anomaly": bool(is_anomaly),
                    "servo_angles": servo_angles.tolist(),
                    "position": position.tolist(),
                    "source": "MLTrainingDatasetGenerator",
                    "metadata": {
                        "baseline_mean": self.baseline_mean,
                        "baseline_std": self.baseline_std,
                        "sample_id": sample_counter,
                        "n_servos": self.n_servos,
                        "servo_origin": self.servo_origin.tolist(),
                        "servo_ceiling": self.servo_ceiling.tolist(),
                        "position_index": int(position_idx),
                        "sample_per_position": sample_per_pos + 1,
                    }
                }

                samples.append(sample)

        dataset = {
            "dataset_metadata": {
                "total_samples": len(samples),
                "anomaly_samples": anomaly_count,
                "normal_samples": len(samples) - anomaly_count,
                "anomaly_rate": anomaly_count / len(samples) if len(samples) > 0 else 0,
                "n_channels": self.n_channels,
                "n_servos": self.n_servos,
                "total_positions": self.total_positions,
                "training_samples": 0,  
                "inference_samples_per_position": self.inference_samples_per_position,
                "generated_timestamp": datetime.now().isoformat(),
                "source": "MLTrainingDatasetGenerator",
                "config_used": bool(self.config),
                "dataset_type": "inference"
            },
            "training_samples": samples,  
        }

        logger.info(f"Inference dataset generation completed:")
        logger.info(f"  Total samples: {len(samples)}")
        logger.info(f"  Normal samples: {len(samples) - anomaly_count}")
        logger.info(f"  Anomaly samples: {anomaly_count}")
        logger.info(
            f"  Samples per position: {self.inference_samples_per_position}")
        logger.info(
            f"  Actual anomaly rate: {anomaly_count / len(samples) * 100:.2f}%")

        return dataset

    def generate_both_datasets(self, training_anomaly_rate: float = 0.05, inference_anomaly_rate: float = 0.0) -> tuple:
        """Generate both training and inference datasets.

        Args:
            training_anomaly_rate (float): Anomaly rate for training data
            inference_anomaly_rate (float): Anomaly rate for inference data

        Returns:
            tuple: (training_dataset, inference_dataset)
        """
        logger.info("Generating both training and inference datasets...")

        training_dataset = self.generate_complete_dataset(
            anomaly_rate=training_anomaly_rate)
        inference_dataset = self.generate_inference_dataset(
            anomaly_rate=inference_anomaly_rate)

        return training_dataset, inference_dataset

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

    def save_both_datasets(self, training_dataset: Dict, inference_dataset: Dict,
                           training_file: str = None, inference_file: str = None):
        """Save both training and inference datasets to JSON files.

        Args:
            training_dataset (Dict): Training dataset dictionary
            inference_dataset (Dict): Inference dataset dictionary
            training_file (str, optional): Training file path
            inference_file (str, optional): Inference file path
        """
        if training_file is None:
            training_file = self.output_file

        if inference_file is None:
            base_name = os.path.splitext(training_file)[0]
            extension = os.path.splitext(training_file)[1]
            inference_file = f"{base_name.replace('training', 'inference')}{extension}"

            if 'training' not in base_name:
                inference_file = f"{base_name}_inference{extension}"

        self.save_dataset(training_dataset, training_file)

        logger.info(f"Saving inference dataset...")
        self.save_dataset(inference_dataset, inference_file)

        logger.info(f"Both datasets saved:")
        logger.info(f"  Training: {training_file}")
        logger.info(f"  Inference: {inference_file}")


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
        help="Anomaly rate for training data (0.0-1.0)"
    )

    parser.add_argument(
        "--inference-anomaly-rate", type=float, default=0.0,
        help="Anomaly rate for inference data (0.0-1.0)"
    )

    parser.add_argument(
        "--channels", type=int, default=5,
        help="Number of EEG channels to simulate"
    )

    parser.add_argument(
        "--generate-inference",
        action="store_true",
        help="Generate inference dataset in addition to training dataset"
    )

    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Generate only inference dataset"
    )

    parser.add_argument(
        "--inference-output",
        help="Output path for inference dataset (auto-generated if not specified)"
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
        parser.error("Training anomaly rate must be between 0.0 and 1.0")

    if not (0.0 <= args.inference_anomaly_rate <= 1.0):
        parser.error("Inference anomaly rate must be between 0.0 and 1.0")

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

        if args.inference_only:
            logger.info("Generating inference dataset only...")
            inference_dataset = generator.generate_inference_dataset(
                anomaly_rate=args.inference_anomaly_rate)

            inference_file = args.inference_output or args.output.replace(
                'training', 'inference')
            generator.save_dataset(inference_dataset, inference_file)

        elif args.generate_inference:
            logger.info("Generating both training and inference datasets...")
            training_dataset, inference_dataset = generator.generate_both_datasets(
                training_anomaly_rate=args.anomaly_rate,
                inference_anomaly_rate=args.inference_anomaly_rate
            )

            generator.save_both_datasets(
                training_dataset,
                inference_dataset,
                training_file=args.output,
                inference_file=args.inference_output
            )

        else:
            logger.info("Generating training dataset only...")
            dataset = generator.generate_complete_dataset(
                anomaly_rate=args.anomaly_rate)
            generator.save_dataset(dataset)

        logger.info("Dataset generation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Failed to generate training dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
