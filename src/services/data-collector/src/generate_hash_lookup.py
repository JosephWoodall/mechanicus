import json
import numpy as np

def generate_hash_lookup(step_size, precision, output_file):
    # Calculate the number of positions based on the step size
    positions = np.arange(0, 180 + step_size, step_size)
    hash_lookup = {}

    for x in positions:
        for y in positions:
            for z in positions:
                # Create a hash key based on the position
                key = (round(x, precision), round(y, precision), round(z, precision))
                hash_lookup[key] = (x, y, z)

    # Write the hash lookup to a JSON file
    with open(output_file, 'w') as f:
        json.dump(hash_lookup, f, indent=4)

if __name__ == "__main__":
    # Configuration parameters
    step_size = 5
    precision = 6
    output_file = "hash_to_servo_lookup.json"

    generate_hash_lookup(step_size, precision, output_file)