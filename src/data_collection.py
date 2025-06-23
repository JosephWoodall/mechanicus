import numpy 
import json
import hashlib
import random
import datetime

class ServoAngleGenerator:
    
    def __init__(self, n_servos=3, origin=[0,0,0], ceiling=[180,180,180], total_positions=100):
        """Generates discrete servo angle combinations that map to 3D cartesian positions.
        
        Each servo has an angle range from origin to ceiling. The combination of all servo angles 
        represents a position in 3D space. This class generates a specified number of discrete 
        positions that the prosthetic arm can move to.

        Args:
            n_servos (int, optional): number of servos in the arm. Defaults to 3.
            origin (list, optional): starting angles for each servo. Defaults to [0,0,0].
            ceiling (list, optional): maximum angles for each servo. Defaults to [180,180,180].
            total_positions (int, optional): total number of unique servo positions to generate. Defaults to 100.
        """
        self.n_servos = n_servos 
        self.total_positions = total_positions
        self.origin = numpy.array(origin if origin is not None else [0] * n_servos)
        self.ceiling = numpy.array(ceiling if ceiling is not None else [180] * n_servos)
        
        # Calculate steps per servo to generate enough combinations
        self.steps_per_servo = max(3, int(numpy.ceil(total_positions ** (1 / n_servos))))
        
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
        
        # Always select exactly total_positions combinations
        if len(combinations) > self.total_positions:
            numpy.random.seed(None)  # Use current time for randomness
            indices = numpy.random.choice(len(combinations), self.total_positions, replace=False)
            combinations = combinations[indices]
        elif len(combinations) < self.total_positions:
            # If we don't have enough combinations, generate more with random variations
            additional_needed = self.total_positions - len(combinations)
            random_variations = []
            
            for _ in range(additional_needed):
                # Create random variations within the servo ranges
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
        numpy.random.seed(None)  # Use current time for randomness
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
        hash_value = hashlib.md5(position_str.encode()).hexdigest()[:12]  # Use first 12 characters
        return hash_value
    
    def generate_complete_dataset(self, n_eeg_channels, samples_per_position=1, mean=0.0, std=1.0, 
                                 total_samples=None, is_inference_data="n"):
        """Generate a complete dataset with eeg data, servo angles, and positions for each eeg sample.
        
        Args:
            n_eeg_channels (int): number of eeg channels.
            samples_per_position (int, optional): number of eeg samples per servo position. Defaults to 1.
            mean (float, optional): mean for eeg data generation. Defaults to 0.0.
            std (float, optional): standard deviation for eeg data generation. Defaults to 1.0.
            total_samples (int, optional): exact total number of samples to generate. 
                                         If specified, overrides samples_per_position calculation.
            is_inference_data (str, optional): "y" to return only first sample, "n" for all samples.
            
        Returns:
            dict: Dataset with samples and metadata.
        """
        
        servo_angles, positions = self.generate_position_mappings()
        
        if total_samples is not None:
            # Calculate how many samples per position we need to reach total_samples
            calculated_samples_per_position = max(1, total_samples // self.total_positions)
            remaining_samples = total_samples % self.total_positions
            
            # Repeat servo angles and positions
            servo_angles = numpy.repeat(servo_angles, calculated_samples_per_position, axis=0)
            positions = numpy.repeat(positions, calculated_samples_per_position, axis=0)
            
            # Add remaining samples if needed
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
        
        hash_to_servo_lookup = {}
        hash_to_position_lookup = {}
        
        position_hashes = []
        for i in range(n_total_samples):
            position_hash = self.position_to_hash(positions[i]) 
            position_hashes.append(position_hash)
            if position_hash not in hash_to_servo_lookup:  # Only store unique combinations
                hash_to_servo_lookup[position_hash] = servo_angles[i].tolist()
                hash_to_position_lookup[position_hash] = positions[i].tolist()
        
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
            'hash_to_servo_lookup': hash_to_servo_lookup,
            'hash_to_position_lookup': hash_to_position_lookup
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
        print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    
    # Example 1: Specify total positions in constructor
    generator = ServoAngleGenerator(
        n_servos=3,
        origin=[0, 0, 0],
        ceiling=[180, 180, 180],
        total_positions=200  # This will create 200 unique servo positions
    )
    
    print(f"Generator configured for {generator.total_positions} unique positions")
    
    # Example 2: Generate dataset with specific total sample count
    training_dataset = generator.generate_complete_dataset(
        n_eeg_channels=5,
        total_samples=1000,  # Exactly 1000 samples total
        mean=0.0,
        std=1.0,
        is_inference_data="n"
    )
    generator.save_dataset_to_json(training_dataset, 'training_data.json')
    
    # Generate inference dataset (single sample)
    inference_dataset = generator.generate_complete_dataset(
        n_eeg_channels=5,
        samples_per_position=1,
        mean=0.0,
        std=1.0,
        is_inference_data="y"
    )
    generator.save_dataset_to_json(inference_dataset, 'inference_data.json')
