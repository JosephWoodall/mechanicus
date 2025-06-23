import numpy 
import json

class ServoAngleGenerator:
    
    def __init__(self, n_servos = 3, origin=[0,0,0], ceiling = [180,180,180]):
        """Generates discrete servo angle combinations that map to 3D cartesian positions.
        
        Each servo has an angle range from origin to ceiling. The combination of all servo angles 
        represents a position in 3D space. This class generates a limited set of discrete position that the 
        prostethic arm can move to.

        Args:
            n_servos (int, optional): number of servos in the arm. Defaults to 3.
            origin (list, optional): starting angles for each servo. Defaults to [0,0,0].
            ceiling (list, optional): maximum angles for each servo. Defaults to [180,180,180].
        """
        self.n_servos = n_servos 
        self.n_positions = 5 * n_servos
        self.origin = numpy.array(origin if origin is not None else [0] * n_servos)
        self.ceiling = numpy.array(ceiling if ceiling is not None else [180] * n_servos)
        
        self.steps_per_servo = int(numpy.ceil(self.n_positions ** (1 / n_servos)))
        
    def generate_servo_combinations(self):
        """Generate discrete servo angle combinations within the specified range.
        Returns:
            numpy.ndarray: Array of shape (n_positions, n_servos) containing the servo angles.
        """
        
        servo_angles = []
        for i in range(self.n_servos):
            angles = numpy.linspace(self.origin[i], self.ceiling[i], self.steps_per_servo)
            servo_angles.append(angles)
        
        angle_grids = numpy.meshgrid(*servo_angles, indexing = 'ij')
        
        combinations = numpy.column_stack([grid.flatten() for grid in angle_grids])
        
        if len(combinations) > self.n_positions:
            indices = numpy.random.choice(len(combinations), self.n_positions, replace=False)
            combinations = combinations[indices]
            
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
        
    def generate_simulated_eeg_data(self, n_rows, n_eeg_channels, mean = 0.0, std = 1.0):
        """Generate simulated EEG data for the specified number of rows and channels.
        
        Args:
            n_rows (int): Number of rows (samples) to generate.
            n_eeg_channels (int): Number of EEG channels.
            mean (float, optional): Mean of the normal distribution. Defaults to 0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 1.0.
            
        Returns:
            numpy.ndarray: Simulated EEG data of shape (n_rows, n_eeg_channels).
        """
        return numpy.random.normal(mean, std, size=(n_rows, n_eeg_channels))
        
    def generate_position_mappings(self):
        """Generate servo angle combinations and their corresponding 3D cartesian positions.
        
        Returns:
            tuple: (servo_angles, cartesian_positions)
                - servo_angles: array of shape (n_positions, n_servos)
                - cartesian_positions: array of shape (n_positions, 3)
        """
        servo_combinations = self.generate_servo_combinations()
        positions = numpy.array([
            self.angles_to_cartesian_position(angles) for angles in servo_combinations
        ])
        
        return servo_combinations, positions 
    
    def generate_complete_dataset(self, n_eeg_channels, samples_per_position = 1, mean = 0.0, std = 1.0):
        """Generate a complete dataset with eeg data, servo angles, and positions for each eeg sample.
        

        Args:
            n_eeg_channels (int): number of eeg channels.
            samples_per_position (int, optional): number of eeg samples per servo position. Defaults to 1.
            mean (float, optional): mean for eeg data generation. Defaults to 0.0.
            std (float, optional): standard deviation for eeg data generation. Defaults to 1.0.
            
        Returns:
            dict: Dictionary containing:
                - 'servo_angles': array of shape (n_total_samples, n_servos)
                - 'positions': array of shape (n_total_samples, 3)
                - 'eeg_data': array of shape (n_total_samples, n_eeg_channels)
                - 'metadata': dict with dataset information
        """
        
        servo_angles, positions = self.generate_position_mappings()
        
        if samples_per_position < 1:
            servo_angles = numpy.repeat(servo_angles, samples_per_position, axis = 0)
            positions = numpy.repeat(positions, samples_per_position, axis = 0)
            
        n_total_samples = len(servo_angles)
        
        eeg_data = self.generate_simulated_eeg_data(
            n_total_samples, 
            n_eeg_channels,
            mean,
            std
        )
        
        metadata = {
            'n_servos':self.n_servos,
            'n_eeg_channels':n_eeg_channels,
            'samples_per_position':samples_per_position,
            'total_samples':n_total_samples,
            'unique_positions': len(servo_angles) // samples_per_position,
            'eeg_mean': mean,
            'eeg_std': std,
            'servo_origin': self.origin.tolist(),
            'servo_ceiling': self.ceiling.tolist(),
            'collection_method': 'simulated'
        }
        
        dataset = {}
        
        for i in range(n_total_samples):
            unique_id = f'sample_{i:04d}'
            dataset[unique_id] = {
                'servo_angles': servo_angles[i].tolist(),
                'position': positions[i].tolist(),
                'eeg_data': eeg_data[i].tolist()
            }
        
        dataset["metadata"] = metadata

        filename = "final_output_example_of_servo_eeg_dataset.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"Dataset saved to {filename}")
        except Exception as e:
            print(f"Error saving dataset to file: {e}")
            
        dataset_return = {
        'servo_angles': servo_angles,
        'positions': positions,
        'eeg_data': eeg_data,
        'metadata': metadata
        }
    
        return dataset_return

if __name__ == "__main__":
    
    generator = ServoAngleGenerator(
        n_servos = 3,
        origin=[0, 0, 0],
        ceiling=[180, 180, 180]
    )
    
    # Generate complete dataset as dictionary
    dataset = generator.generate_complete_dataset(
        n_eeg_channels=5,
        samples_per_position=5,
        mean=0.0,
        std=1.0
    )
    print("\n" + "-" * 50 + "\n")
    print("Generated complete dataset:")
    print(f"Servo angles shape: {dataset['servo_angles'].shape}")
    print(f"Positions shape: {dataset['positions'].shape}")
    print(f"EEG data shape: {dataset['eeg_data'].shape}")
    
    print("\nMetadata:")
    for key, value in dataset['metadata'].items():
        print(f"  {key}: {value}")
    
    print("\nFirst 5 servo angles:")
    print(dataset['servo_angles'][:5])
    
    print("\nFirst 5 positions:")
    print(dataset['positions'][:5])
    
    print("\nFirst 3 EEG samples (first 5 channels):")
    print(dataset['eeg_data'][:3, :5])   
    
    print("\n" + "-" * 50 + "\n") 
