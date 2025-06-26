import logging, warnings

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="mechanicus_data_collection.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

import numpy 
import json
import hashlib
import random
import datetime
import yaml
from pathlib import Path

class ServoAngleGenerator:
    
    def __init__(self, config=None):
        """Generates discrete servo angle combinations that map to 3D cartesian positions.
        
        Each servo has an angle range from origin to ceiling. The combination of all servo angles 
        represents a position in 3D space. This class generates a specified number of discrete 
        positions that the prosthetic arm can move to.

        Args:
            config (dict, optional): Configuration dictionary. If None, loads from mechanicus_run_configuration.yaml.
        """
        if config is None:
            config = self.load_config()
        
        servo_config = config.get('servo_config', {})
        self.n_servos = servo_config.get('n_servos', 3)
        self.total_positions = servo_config.get('total_positions', 100)
        self.origin = numpy.array(servo_config.get('origin', [0, 0, 0]))
        self.ceiling = numpy.array(servo_config.get('ceiling', [180, 180, 180]))
        
        self.steps_per_servo = max(3, int(numpy.ceil(self.total_positions ** (1 / self.n_servos))))
        
        logging.info(f"ServoAngleGenerator initialized:")
        logging.info(f"  - Servos: {self.n_servos}")
        logging.info(f"  - Origin: {self.origin.tolist()}")
        logging.info(f"  - Ceiling: {self.ceiling.tolist()}")
        logging.info(f"  - Total positions: {self.total_positions}")
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open("mechanicus_run_configuration.yaml", 'r') as f:
                config = yaml.safe_load(f)
                logging.info("Loaded configuration from mechanicus_run_configuration.yaml")
                return config
        except Exception as e:
            logging.info(f"Warning: Could not load mechanicus_run_configuration.yaml: {e}")
            logging.info("Using default configuration values")
            return {}
        
    def generate_servo_combinations(self):
        """Generate discrete servo angle combinations within the specified range.
        Returns:
            numpy.ndarray: Array of shape (total_positions, n_servos) containing the servo angles.
        """
        
        servo_angles = []
        for i in range(self.n_servos):
            angles = numpy.linspace(self.origin[i], self.ceiling[i], self.steps_per_servo)
            servo_angles.append(angles)
        
        angle_grids = numpy.meshgrid(*servo_angles, indexing='ij')
        
        combinations = numpy.column_stack([grid.flatten() for grid in angle_grids])
        
        if len(combinations) > self.total_positions:
            numpy.random.seed(None)
            indices = numpy.random.choice(len(combinations), self.total_positions, replace=False)
            combinations = combinations[indices]
        elif len(combinations) < self.total_positions:
            additional_needed = self.total_positions - len(combinations)
            random_variations = []
            
            for _ in range(additional_needed):
                random_combo = []
                for i in range(self.n_servos):
                    random_angle = numpy.random.uniform(self.origin[i], self.ceiling[i])
                    random_combo.append(random_angle)
                random_variations.append(random_combo)
            
            combinations = numpy.vstack([combinations, numpy.array(random_variations)])
            
        return combinations 
    
    def angles_to_cartesian_position(self, servo_angles):
        """Convert servo angles to a 3D cartesian position.
        
        This is a simplified mapping - you may need to adjust based on the specific arm kinematics.

        Args:
            servo_angles (numpy.ndarray): array of servo angles.
            
        Returns:
            numpy.ndarray: 3D cartesian position corresponding to the servo angles [x, y, z].
        """
        
        if len(servo_angles) >= 3:
            theta = numpy.radians(servo_angles[0]) 
            phi = numpy.radians(servo_angles[1])
            r = servo_angles[2] / 180.0 
            
            x = r * numpy.sin(phi) * numpy.cos(theta)
            y = r * numpy.sin(phi) * numpy.sin(theta)
            z = r * numpy.cos(phi)
            
            return numpy.array([x, y, z])
        else:
            position = numpy.zeros(3)
            for i, angle in enumerate(servo_angles[:3]):
                position[i] = angle / 180.0
            return position 
        
    def generate_simulated_eeg_data(self, n_rows, n_eeg_channels, mean=0.0, std=1.0):
        """Generate simulated EEG data for the specified number of rows and channels.
        
        Args:
            n_rows (int): Number of rows (samples) to generate.
            n_eeg_channels (int): Number of EEG channels.
            mean (float, optional): Mean of the normal distribution. Defaults to 0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 1.0.
            
        Returns:
            numpy.ndarray: Simulated EEG data of shape (n_rows, n_eeg_channels).
        """
        numpy.random.seed(None)
        return numpy.random.normal(mean, std, size=(n_rows, n_eeg_channels))
        
    def generate_position_mappings(self):
        """Generate servo angle combinations and their corresponding 3D cartesian positions.
        
        Returns:
            tuple: (servo_angles, cartesian_positions)
                - servo_angles: array of shape (total_positions, n_servos)
                - cartesian_positions: array of shape (total_positions, 3)
        """
        servo_combinations = self.generate_servo_combinations()
        positions = numpy.array([
            self.angles_to_cartesian_position(angles) for angles in servo_combinations
        ])
        
        return servo_combinations, positions 
    
    def position_to_hash(self, position, precision=6):
        """Convert a 3D position to a hash value.

        Args:
            position (numpy.ndarray or list): 3D position [x, y, z].
            precision (int, optional): number of decimal places to round to for consistency. Defaults to 6.
            
        Returns:
            str: Hash value as a string.
        """
        rounded_position = numpy.round(position, precision)
        position_str = f"{rounded_position[0]:.{precision}f}_{rounded_position[1]:.{precision}f}_{rounded_position[2]:.{precision}f}"
        hash_value = hashlib.md5(position_str.encode()).hexdigest()[:12]
        return hash_value
    
    def generate_complete_dataset(self, n_eeg_channels=None, samples_per_position=1, mean=0.0, std=1.0, 
                                 total_samples=None, is_inference_data="n"):
        """Generate a complete dataset with eeg data, servo angles, and positions for each eeg sample.
        
        Args:
            n_eeg_channels (int, optional): number of eeg channels. If None, loads from config.
            samples_per_position (int, optional): number of eeg samples per servo position. Defaults to 1.
            mean (float, optional): mean for eeg data generation. Defaults to 0.0.
            std (float, optional): standard deviation for eeg data generation. Defaults to 1.0.
            total_samples (int, optional): exact total number of samples to generate. 
                                         If specified, overrides samples_per_position calculation.
            is_inference_data (str, optional): "y" to return only first sample, "n" for all samples.
            
        Returns:
            dict: Dataset with samples and metadata (WITHOUT hash lookup tables).
        """
        
        if n_eeg_channels is None:
            config = self.load_config()
            dataset_config = config.get('dataset', {})
            n_eeg_channels = dataset_config.get('n_eeg_channels', 5)
            
        servo_angles, positions = self.generate_position_mappings()
        
        if total_samples is not None:
            calculated_samples_per_position = max(1, total_samples // self.total_positions)
            remaining_samples = total_samples % self.total_positions
            
            servo_angles = numpy.repeat(servo_angles, calculated_samples_per_position, axis=0)
            positions = numpy.repeat(positions, calculated_samples_per_position, axis=0)
            
            if remaining_samples > 0:
                extra_servo = servo_angles[:remaining_samples]
                extra_positions = positions[:remaining_samples]
                servo_angles = numpy.vstack([servo_angles, extra_servo])
                positions = numpy.vstack([positions, extra_positions])
                
            samples_per_position = calculated_samples_per_position
            
        elif samples_per_position > 1:
            servo_angles = numpy.repeat(servo_angles, samples_per_position, axis=0)
            positions = numpy.repeat(positions, samples_per_position, axis=0)
            
        n_total_samples = len(servo_angles)
        
        eeg_data = self.generate_simulated_eeg_data(
            n_total_samples, 
            n_eeg_channels,
            mean,
            std
        )
        
        position_hashes = []
        for i in range(n_total_samples):
            position_hash = self.position_to_hash(positions[i]) 
            position_hashes.append(position_hash)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = random.randint(1000, 9999)
        
        metadata = {
            'n_servos': self.n_servos,
            'n_eeg_channels': n_eeg_channels,
            'samples_per_position': samples_per_position,
            'total_samples': n_total_samples,
            'unique_positions': self.total_positions,
            'eeg_mean': mean,
            'eeg_std': std,
            'servo_origin': self.origin.tolist(),
            'servo_ceiling': self.ceiling.tolist(),
            'collection_method': 'simulated',
            'timestamp': timestamp,
            'dataset_id': dataset_id,
            'hash_lookup_file': 'hash_to_servo_lookup.json'  
        }
        
        dataset = {}
        for i in range(n_total_samples):
            unique_id = f'sample_{i:04d}'
            dataset[unique_id] = {
                'servo_angles': servo_angles[i].tolist(),
                'position': positions[i].tolist(),
                'position_hash': position_hashes[i],
                'eeg_data': eeg_data[i].tolist()
            }
        
        dataset["metadata"] = metadata
        
        if is_inference_data == "y":
            first_sample_key = f'sample_{0:04d}'
            dataset = {
                first_sample_key: dataset[first_sample_key],
                "metadata": metadata
            }
            
        return dataset 
    
    def save_dataset_to_json(self, dataset, filename):
        """Save the dataset to a JSON file.
        
        Args:
            dataset (dict): The dataset to save.
            filename (str): The name of the file to save the dataset to.
        """
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        logging.info(f"Dataset saved to {filename}")

if __name__ == "__main__":
    
    logging.info("=" * 60)
    logging.info("STEP 1: Generating comprehensive hash lookup table...")
    logging.info("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(["python", "src/generate_hash_lookup.py"], 
                              capture_output=True, text=True, check=True)
        logging.info("Hash lookup generation completed successfully!")
        if result.stdout:
            logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.info(f"Error generating hash lookup: {e}")
        logging.info("Stderr:", e.stderr)
    except Exception as e:
        logging.info(f"Could not run hash lookup generator: {e}")
        logging.info("Please run 'python src/generate_hash_lookup.py' manually first")
    
    logging.info("\n" + "=" * 60)
    logging.info("STEP 2: Generating datasets...")
    logging.info("=" * 60)
    
    generator = ServoAngleGenerator()
    config = generator.load_config()
    dataset_config = config.get('dataset', {})
    
    logging.info(f"Dataset configuration:")
    logging.info(f"  - EEG channels: {dataset_config.get('n_eeg_channels', 5)}")
    logging.info(f"  - Training samples: {dataset_config.get('training_samples', 1000)}")
    logging.info(f"  - Inference samples per position: {dataset_config.get('inference_samples_per_position', 1)}")
    
    if not Path("training_data.json").exists():
        logging.info("\nGenerating training dataset...")
        training_dataset = generator.generate_complete_dataset(
            n_eeg_channels=dataset_config.get('n_eeg_channels', 5),
            total_samples=dataset_config.get('training_samples', 1000),
            mean=dataset_config.get('eeg_mean', 0.0),
            std=dataset_config.get('eeg_std', 1.0),
            is_inference_data="n"
        )
        generator.save_dataset_to_json(training_dataset, 'training_data.json')
    else:
        logging.info("Training dataset already exists.")
    
    if not Path("inference_data.json").exists():
        logging.info("\nGenerating inference dataset...")
        inference_dataset = generator.generate_complete_dataset(
            n_eeg_channels=dataset_config.get('n_eeg_channels', 5),
            samples_per_position=dataset_config.get('inference_samples_per_position', 1),
            mean=dataset_config.get('eeg_mean', 0.0),
            std=dataset_config.get('eeg_std', 1.0),
            is_inference_data="y"
        )
        generator.save_dataset_to_json(inference_dataset, 'inference_data.json')
    else:
        logging.info("Inference dataset already exists.")
    
    logging.info("\n" + "=" * 60)
    logging.info("DATA COLLECTION COMPLETE!")
    logging.info("=" * 60)
    logging.info("Files generated:")
    logging.info("  - hash_to_servo_lookup.json (comprehensive lookup table)")
    logging.info("  - training_data.json (clean training data)")
    logging.info("  - inference_data.json (clean inference data)")