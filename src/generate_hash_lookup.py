import numpy 
import json
import hashlib
import datetime
import yaml
from pathlib import Path

class HashLookupGenerator:
    """Generate comprehensive hash-to-servo lookup table using shared config"""
    
    def __init__(self, config=None):
        # Load from config or use defaults
        if config is None:
            config = self.load_config()
        
        servo_config = config.get('servo_config', {})
        self.n_servos = servo_config.get('n_servos', 3)
        self.origin = numpy.array(servo_config.get('origin', [0, 0, 0]))
        self.ceiling = numpy.array(servo_config.get('ceiling', [180, 180, 180]))
        
        hash_config = config.get('hash_lookup', {})
        self.step_size = hash_config.get('step_size', 10)
        self.precision = hash_config.get('precision', 6)
        self.output_file = hash_config.get('output_file', 'hash_to_servo_lookup.json')
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open("mechanicus_run_configuration.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load mechanicus_run_configuration.yaml: {e}")
            return {}
        
    def angles_to_cartesian_position(self, servo_angles):
        """Convert servo angles to 3D cartesian position (same as data_collection.py)"""
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
        
    def position_to_hash(self, position):
        """Convert 3D position to hash (same method as data_collection.py)"""
        rounded_position = numpy.round(position, self.precision)
        position_str = f"{rounded_position[0]:.{self.precision}f}_{rounded_position[1]:.{self.precision}f}_{rounded_position[2]:.{self.precision}f}"
        hash_value = hashlib.md5(position_str.encode()).hexdigest()[:12]
        return hash_value
    
    def generate_comprehensive_lookup(self):
        """Generate comprehensive hash lookup table ensuring FULL coverage"""
        
        print(f"Generating comprehensive hash lookup...")
        print(f"Servo range: {self.origin} to {self.ceiling}")
        print(f"Step size: {self.step_size} degrees")
        
        # Generate ALL servo angle combinations with specified step size
        servo_combinations = []
        
        # Create ranges for each servo
        servo_ranges = []
        for i in range(self.n_servos):
            servo_range = list(range(int(self.origin[i]), int(self.ceiling[i]) + 1, self.step_size))
            servo_ranges.append(servo_range)
            print(f"Servo {i+1}: {len(servo_range)} steps from {self.origin[i]} to {self.ceiling[i]}")
        
        # Generate all combinations
        for servo1 in servo_ranges[0]:
            for servo2 in servo_ranges[1]:
                for servo3 in servo_ranges[2]:
                    servo_combinations.append([float(servo1), float(servo2), float(servo3)])
        
        total_combinations = len(servo_combinations)
        print(f"Generated {total_combinations} servo angle combinations")
        
        # Build lookup dictionaries
        hash_to_servo_lookup = {}
        hash_to_position_lookup = {}
        
        print("Processing combinations and generating hashes...")
        for i, servo_angles in enumerate(servo_combinations):
            # Calculate position from servo angles
            position = self.angles_to_cartesian_position(servo_angles)
            
            # Generate hash from position
            position_hash = self.position_to_hash(position)
            
            # Store in lookup tables (handle duplicates)
            if position_hash not in hash_to_servo_lookup:
                hash_to_servo_lookup[position_hash] = servo_angles
                hash_to_position_lookup[position_hash] = position.tolist()
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{total_combinations} combinations")
        
        # Create lookup file structure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        lookup_data = {
            'metadata': {
                'description': 'Comprehensive hash to servo angle and position lookup table',
                'total_combinations_generated': total_combinations,
                'unique_hashes': len(hash_to_servo_lookup),
                'servo_config': {
                    'n_servos': self.n_servos,
                    'origin': self.origin.tolist(),
                    'ceiling': self.ceiling.tolist()
                },
                'hash_config': {
                    'step_size': self.step_size,
                    'precision': self.precision
                },
                'generated_timestamp': timestamp,
                'generation_method': 'grid_based_comprehensive_from_config'
            },
            'hash_to_servo_lookup': hash_to_servo_lookup,
            'hash_to_position_lookup': hash_to_position_lookup
        }
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(lookup_data, f, indent=2)
        
        print(f"Hash lookup table saved to {self.output_file}")
        print(f"Total unique hashes: {len(hash_to_servo_lookup)}")
        print(f"Coverage: {len(hash_to_servo_lookup)}/{total_combinations} unique positions")
        
        return lookup_data

if __name__ == "__main__":
    generator = HashLookupGenerator()
    generator.generate_comprehensive_lookup()
    print("Hash lookup generation complete!")