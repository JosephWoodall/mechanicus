import logging
import yaml
import warnings
from subprocess import call
import os
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="mechanicus_main.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

def load_config():
    """Load configuration from YAML config file"""
    try:
        with open("mechanicus_run_configuration.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Create comprehensive default configuration
        default_config = {
            'test_mode': True,
            'servo_config': {
                'n_servos': 3,
                'origin': [0, 0, 0],
                'ceiling': [180, 180, 180],
                'total_positions': 200
            },
            'hash_lookup': {
                'step_size': 5,  # Every 5 degrees = 37^3 = 50,653 combinations
                'precision': 6,  # Hash precision for position rounding
                'output_file': "hash_to_servo_lookup.json"
            },
            'dataset': {
                'n_eeg_channels': 5,
                'training_samples': 1000,
                'inference_samples_per_position': 1,
                'eeg_mean': 0.0,
                'eeg_std': 1.0
            }
        }
        
        # Create config file in root directory (not in src/)
        with open("mechanicus_run_configuration.yaml", 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        logging.info("Created default mechanicus_run_configuration.yaml with comprehensive configuration")
        logging.info(f"Default config: {default_config}")
        return default_config
        
    except Exception as e:
        logging.error(f"Error loading config: {e}. Using default configuration.")
        # Return default config even if file creation fails
        return {
            'test_mode': True,
            'servo_config': {
                'n_servos': 3,
                'origin': [0, 0, 0],
                'ceiling': [180, 180, 180],
                'total_positions': 200
            },
            'hash_lookup': {
                'step_size': 5,
                'precision': 6,
                'output_file': "hash_to_servo_lookup.json"
            },
            'dataset': {
                'n_eeg_channels': 5,
                'training_samples': 1000,
                'inference_samples_per_position': 1,
                'eeg_mean': 0.0,
                'eeg_std': 1.0
            }
        }

def main():
    try:
        config = load_config()
        test_mode = config.get('test_mode', True)
        
        logging.debug(f"Current working directory: {os.getcwd()}")
        logging.debug(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
        logging.info(f"Running in {'TEST' if test_mode else 'HARDWARE'} mode")
        
        # Log configuration details
        servo_config = config.get('servo_config', {})
        hash_config = config.get('hash_lookup', {})
        dataset_config = config.get('dataset', {})
        
        logging.info(f"Servo configuration: {servo_config}")
        logging.info(f"Hash lookup configuration: {hash_config}")
        logging.info(f"Dataset configuration: {dataset_config}")
        
        # Check for hash lookup file
        hash_lookup_file = hash_config.get('output_file', 'hash_to_servo_lookup.json')
        if not Path(hash_lookup_file).exists():
            logging.info(f"No {hash_lookup_file} found. Running the lookup generation script...")
            try:
                call(["python", "src/generate_hash_lookup.py"])
                logging.info("...Hash Lookup Generation Complete.")
            except Exception as e:
                logging.error(f"Error running the Hash Lookup Generation Script: {e}")
        
        # Check for training data
        if not Path("training_data.json").exists():
            logging.info("No training_data.json found. Running the data collection script...")
            try:
                call(["python", "src/data_collection.py"])
                logging.info("...Data Collection Complete.")
            except Exception as e:
                logging.error(f"Error running the Data Collection Script: {e}")
        
        # Check for inference data
        if not Path("inference_data.json").exists():
            logging.info("No inference_data.json found. Running the data collection script...")
            try:
                call(["python", "src/data_collection.py"])
                logging.info("...Inference Data Generation Complete.")
            except Exception as e:
                logging.error(f"Error running the Data Collection Script: {e}")

        # Check for ML model
        try:
            ml_model = Path("inference_model.pkl")
            if not ml_model.exists():
                try:
                    logging.info(
                        "No Loaded Machine Learning Model Exists. Running Training Process Now and Creating Inference Model..."
                    )
                    call(["python", "src/train.py"])
                    logging.info("...Training Process Complete.")
                except Exception as e:
                    logging.error(f"Error running the Training Process: {e}")
            elif ml_model.exists():
                logging.info(
                    "Existing Machine Learning Model Found. Proceeding with Action Sequence."
                )
        except Exception as e:
            logging.error(
                f"Something went wrong when interacting with the Training Process. Please confirm if the model exists under 'inference_model.pkl' and try again."
            )

        # Run Action Sequence
        logging.info(f"Running Action Sequence in {'TEST' if test_mode else 'HARDWARE'} mode...")
        try:
            if test_mode:
                call(["python", "src/action.py", "--test"])
            else:
                call(["python", "src/action.py"])
        except Exception as e:
            logging.error(f"Error running the Action Sequence: {e}")
        logging.info("...Action Sequence Terminated.")
        
    except Exception as e:
        logging.error(f"Error running the main file: {e}")

if __name__ == "__main__":
    logging.info("-" * 200)
    logging.info("----------STARTING Mechanicus----------")
    logging.info("-" * 200)

    main()
    
    logging.info("-" * 200)
    logging.info(f"----------TERMINATING Mechanicus---------")
    logging.info("-" * 200)