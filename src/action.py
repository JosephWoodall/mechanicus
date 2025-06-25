import sys
import logging
import json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="mechanicus_action.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas
import time
import os
from pathlib import Path

# Only import pyfirmata2 if not in test mode
if "--test" not in sys.argv and '-t' not in sys.argv:
    try:
        import pyfirmata2
    except ImportError:
        logging.warning("pyfirmata2 not available - hardware mode may not work")

from train import DataCollector, PreprocessData, Phase

class ServoController:
    """This class handles servo initialization and positioning"""
    
    def __init__(self, test_mode = False):
        self.test_mode = test_mode
        self.board = None
        self.servos = []  
        self.current_positions = [0, 0, 0]  
    
    def initialize_servos(self):
        """Initialize servo connections and move to starting positions"""
        try:
            if self.test_mode:
                logging.info("TEST MODE: Simulating servo initialization")
                self.servos = ["simulated_servo_1", "simulated_servo_2", "simulated_servo_3"]
                logging.info(f"TEST MODE: Servos initialized to starting positions: {self.current_positions}")
                return True
            elif self.test_mode == False:
                port = pyfirmata2.Arduino.AUTODETECT
                self.board = pyfirmata2.Arduino(port)
                
                time.sleep(2) 
                
                servo_pins = ["d:9:s", "d:10:s", "d:11:s"]
                for pin in servo_pins:
                    servo = self.board.get_pin(pin)
                    self.servos.append(servo)
                
                self.move_to_positions([0, 0, 0])
                logging.info(f"Servos initialized to starting positions: {self.current_positions}")
                return True
        except Exception as e:
            logging.error(f"Failed to initialize servos: {e}")
            return False
    
    def move_to_positions(self, positions):
        """Move servos to specific positions"""
        if self.test_mode: 
            logging.info(f"TEST MODE: Simulating servo movement to positions: {positions}")
            for i, position in enumerate(positions):
                if i < len(self.current_positions):
                    self.current_positions[i] = position
            logging.debug(f"TEST MODE: Current positions updated to: {self.current_positions}")
            return
        if len(self.servos) >= len(positions):
            for i, position in enumerate(positions):
                if i < len(self.servos):
                    self.servos[i].write(position)
                    self.current_positions[i] = position
            logging.debug(f"Servos moved to positions: {self.current_positions}")
    
    def reset_to_origin(self):
        """Reset all servos to position 0"""
        if self.test_mode:
            logging.info("TEST MODE: Simulating reset to origin positions")
            self.current_positions = [0, 0, 0]
            logging.debug(f"TEST MODE: Current positions reset to: {self.current_positions}")
            return
        self.move_to_positions([0, 0, 0])
        logging.info(f"Servos reset to origin positions: {self.current_positions}")
    
    def get_current_positions(self):
        """Get current servo positions"""
        return self.current_positions
    
    def cleanup(self):
        """Clean up board connection"""
        if self.board and not self.test_mode:
            try:
                self.board.exit()
                logging.info("Board connection cleaned up successfully.")
            except Exception as e:
                logging.error(f"Error cleaning up board connection: {e}")

class HashToServoLookup:
    """Class to handle hash to servo angle lookup from dedicated hash_to_servo_lookup.json file"""
    
    def __init__(self, lookup_file="hash_to_servo_lookup.json"):
        """Initialize lookup tables from dedicated hash lookup file"""
        self.hash_to_servo = {}
        self.hash_to_position = {}
        self.lookup_file = lookup_file
        self.load_lookup_tables()
    
    def load_lookup_tables(self):
        """Load lookup tables from dedicated hash_to_servo_lookup.json file"""
        try:
            lookup_path = Path(self.lookup_file)
            
            if not lookup_path.exists():
                logging.warning(f"Hash lookup file {self.lookup_file} not found!")
                logging.info("Attempting to generate hash lookup file...")
                self._generate_lookup_file()
                
                # Try loading again after generation
                if not lookup_path.exists():
                    logging.error("Failed to generate hash lookup file")
                    self._fallback_to_inference_data()
                    return
            
            with open(self.lookup_file, 'r') as f:
                data = json.load(f)
            
            # Load from the hash lookup file structure
            self.hash_to_servo = data.get('hash_to_servo_lookup', {})
            self.hash_to_position = data.get('hash_to_position_lookup', {})
            
            metadata = data.get('metadata', {})
            total_mappings = metadata.get('total_mappings', len(self.hash_to_servo))
            step_size = metadata.get('servo_step_size', metadata.get('step_size', 'unknown'))
            
            logging.info(f"Loaded {len(self.hash_to_servo)} hash-to-servo mappings from {self.lookup_file}")
            logging.info(f"Total mappings in file: {total_mappings}")
            logging.info(f"Lookup table step size: {step_size}")
            
            if len(self.hash_to_servo) == 0:
                logging.error("No hash mappings found in lookup file!")
                self._fallback_to_inference_data()
                
        except Exception as e:
            logging.error(f"Error loading hash lookup file {self.lookup_file}: {e}")
            logging.info("Falling back to inference data...")
            self._fallback_to_inference_data()
    
    def _generate_lookup_file(self):
        """Generate the hash lookup file if missing"""
        try:
            import subprocess
            logging.info("Generating hash lookup table...")
            result = subprocess.run(["python", "src/generate_hash_lookup.py"], 
                                  capture_output=True, text=True, check=True)
            logging.info("Hash lookup file generated successfully")
            if result.stdout:
                logging.debug(f"Generation output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate hash lookup file: {e}")
            if e.stderr:
                logging.error(f"Generation error: {e.stderr}")
        except Exception as e:
            logging.error(f"Could not run hash lookup generator: {e}")
    
    def _fallback_to_inference_data(self):
        """Fallback: Load from inference data if lookup file fails"""
        try:
            logging.info("Attempting to load hash data from inference_data.json as fallback...")
            with open("inference_data.json", 'r') as f:
                data = json.load(f)
            
            # Check metadata first
            metadata = data.get('metadata', {})
            self.hash_to_servo = metadata.get('hash_to_servo_lookup', {})
            self.hash_to_position = metadata.get('hash_to_position_lookup', {})
            
            # If not in metadata, create from samples
            if not self.hash_to_servo:
                logging.info("No hash lookup in metadata, creating from samples...")
                self._create_lookup_from_samples(data)
                
            logging.info(f"Fallback: Loaded {len(self.hash_to_servo)} hash mappings from inference data")
            
        except Exception as e:
            logging.error(f"Fallback also failed: {e}")
            self.hash_to_servo = {}
            self.hash_to_position = {}
    
    def _create_lookup_from_samples(self, data):
        """Create lookup from sample data if not in metadata"""
        for key, sample in data.items():
            if key != 'metadata' and isinstance(sample, dict):
                hash_val = sample.get('position_hash')
                servo_angles = sample.get('servo_angles')
                position = sample.get('position')
                
                if hash_val and servo_angles and position:
                    if hash_val not in self.hash_to_servo:
                        self.hash_to_servo[hash_val] = servo_angles
                        self.hash_to_position[hash_val] = position
    
    def get_servo_angles_from_hash(self, position_hash):
        """Convert position hash to servo angles with intelligent fallback"""
        servo_angles = self.hash_to_servo.get(position_hash, None)
        
        if servo_angles is None:
            logging.warning(f"Hash {position_hash} not found in lookup table")
            
            # Try to find closest match
            closest_hash = self._find_closest_hash(position_hash)
            if closest_hash:
                logging.info(f"Using closest match hash: {closest_hash}")
                servo_angles = self.hash_to_servo[closest_hash]
            else:
                # Generate fallback based on hash
                servo_angles = self._generate_fallback_angles(position_hash)
                logging.info(f"Generated fallback servo angles: {servo_angles}")
                
        return servo_angles
    
    def get_position_from_hash(self, position_hash):
        """Convert position hash to 3D coordinates"""
        position = self.hash_to_position.get(position_hash, None)
        if position is None:
            logging.warning(f"No position found for hash: {position_hash}")
            # Try closest match
            closest_hash = self._find_closest_hash(position_hash)
            if closest_hash:
                position = self.hash_to_position.get(closest_hash, [0, 0, 0])
            else:
                position = [0, 0, 0]
        return position
    
    def _find_closest_hash(self, target_hash):
        """Find closest matching hash using string similarity"""
        if not self.hash_to_servo:
            return None
            
        available_hashes = list(self.hash_to_servo.keys())
        
        # Simple approach: find hash with most matching characters
        best_match = None
        best_score = 0
        
        for hash_candidate in available_hashes:
            # Count matching characters at same positions
            score = sum(1 for a, b in zip(target_hash, hash_candidate) if a == b)
            if score > best_score:
                best_score = score
                best_match = hash_candidate
        
        # Only return if we have reasonable similarity (at least 50% match)
        if best_score >= len(target_hash) * 0.5:
            return best_match
        return None
    
    def _generate_fallback_angles(self, position_hash):
        """Generate servo angles from hash when no match found"""
        # Use hash to generate deterministic but reasonable servo angles
        try:
            hash_int = int(position_hash[:8], 16) % (2**32)  # Use first 8 chars
        except ValueError:
            hash_int = hash(position_hash) % (2**32)
        
        # Generate angles in valid servo range using hash as seed
        import random
        random.seed(hash_int)
        
        servo_angles = [
            random.randint(0, 180),
            random.randint(0, 180), 
            random.randint(0, 180)
        ]
        
        return servo_angles
    
    def list_available_hashes(self):
        """List all available hashes for debugging"""
        available_hashes = list(self.hash_to_servo.keys())
        logging.info(f"Available hashes ({len(available_hashes)} total): {available_hashes[:5]}...")
        return available_hashes 
    
    def test_lookup(self):
        """Test the lookup functionality"""
        logging.info("Testing HashToServoLookup functionality...")
        available_hashes = self.list_available_hashes()
        
        if not available_hashes:
            logging.error("No hashes available for testing.")
            return False
        
        for i, test_hash in enumerate(available_hashes[:5]):
            servo_angles = self.get_servo_angles_from_hash(test_hash)
            position = self.get_position_from_hash(test_hash)
            
            logging.info(f"Test {i+1}: Hash '{test_hash}'")
            logging.info(f"  -> Servo angles: {servo_angles}")
            logging.info(f"  -> 3D position: {position}")
        
        return True

class Action:
    """This class contains functions that leverage the output of the Inference class and move the servo to the appropriate position"""
    
    _lookup = None
    
    @classmethod
    def _get_lookup(cls):
        """Get or create the hash lookup instance using hash_to_servo_lookup.json"""
        if cls._lookup is None:
            cls._lookup = HashToServoLookup("hash_to_servo_lookup.json")
        return cls._lookup
    
    @staticmethod
    def _ensure_inference_data_exists():
        """Check if inference data exists, generate if needed"""
        inference_file = Path("inference_data.json")
        
        if not inference_file.exists() or not inference_file.is_file():
            logging.info("Inference data file not found. Generating inference dataset...")
            try:
                # Import data collection module
                from data_collection import ServoAngleGenerator
                
                # Generate inference data using config
                generator = ServoAngleGenerator()
                
                inference_dataset = generator.generate_complete_dataset(
                    samples_per_position=1,
                    is_inference_data="y"
                )
                
                generator.save_dataset_to_json(inference_dataset, 'inference_data.json')
                logging.info("Inference dataset generated successfully.")
                return True
                
            except Exception as e:
                logging.error(f"Error generating inference data: {e}")
                return False
        else:
            logging.info("Inference data file found.")
            return True

    def __load_model_for_inference_from_file(
        filename: str = "inference_model.pkl",
    ) -> object:
        logging.info(f"Loading saved model {filename} for inference...")
        try:
            with open(rf"{filename}", "rb") as f:
                data_from_pickle = pickle.load(f)
                loaded_model = data_from_pickle["model"]
            logging.info(f"...Loaded saved model {filename} for inference.")
        except Exception as e:
            logging.error(
                f"An error occured when attempting to load saved model {filename} for inference: {e}"
            )

        return loaded_model
    
    def __collect_inference_data():
        if not Action._ensure_inference_data_exists():
            logging.error("Failed to ensure inference data exists.")
            return None
        inference_data = DataCollector.load_servo_eeg_data("inference_data.json")
        return inference_data

    def __perform_inference():
        response_variable_production = "position_hash" 
        inference_data = Action.__collect_inference_data()
        
        if inference_data is None:
            logging.error("No inference data available for inference.")
            return None
            
        preprocessed_inference_data = PreprocessData.preprocess_data(
            inference_data, response_variable_production, Phase.INFERENCE
        )
        preprocessed_inference_x = preprocessed_inference_data[0]
        loaded_model = Action.__load_model_for_inference_from_file()
        logging.info("Performing inference using loaded model...")
        try:
            predictions = loaded_model.predict(preprocessed_inference_x)
            predictions_dataframe = pandas.DataFrame({"predictions": predictions}) 
            logging.info("...Inference using loaded model complete.")
            inferenced_action = predictions_dataframe.iloc[0, 0]
            return inferenced_action
        except Exception as e:
            logging.error(
                f"An error occured when performing inference using loaded model: {e}"
            )
            return None

    def __get_inference_value():
        inference_value = Action.__perform_inference()
        logging.info(f"Inferenced Action from Model: {inference_value}")
        return inference_value

    def perform_action(servo_controller: ServoController) -> None:
        """perform_action function is the location in which the machine learning algorithm interacts with hardware.
        """
        logging.info("-" * 100)
        logging.debug("NEW ACTION")
        logging.info("-" * 100)
        try:
            
            logging.debug(f"Current Servo Positions: {servo_controller.get_current_positions()}")

           
            predicted_hash = Action.__get_inference_value()
            logging.info(f"Predicted Hash from Model: {predicted_hash}")
            
            if predicted_hash is None:
                logging.error("No valid prediction received")
                return
            
            
            lookup = Action._get_lookup()
            
            
            target_servo_angles = lookup.get_servo_angles_from_hash(predicted_hash)
            target_position = lookup.get_position_from_hash(predicted_hash)
            
            logging.info(f"Target servo angles: {target_servo_angles}")
            logging.info(f"Target 3D position: {target_position}")
            
            
            servo_controller.move_to_positions(target_servo_angles)
            
            logging.info(f"Final Servo Positions: {servo_controller.get_current_positions()}")
            
        except Exception as e:
            logging.error(f"Something went wrong when attempting to perform an action: {e}")
            logging.error("Exiting action phase.")

if __name__ == "__main__":
    test_mode = "--test" in sys.argv or '-t' in sys.argv
    
    logging.info("-" * 100)
    if test_mode:
        logging.info("Starting Mechanicus Action Pipeline (TEST MODE)")
        print("Running in TEST MODE: Simulating servo movements without hardware.")
    else:
        logging.info("Starting Mechanicus Action Pipeline")
        print("Running in NORMAL MODE: Interacting with hardware.")
    logging.info("-" * 100)
    logging.info("-" * 50)
    logging.debug("This is the Action Sequence. This is where the machine learning model interacts with the hardware.")
    logging.info("-" * 50)

    # Ensure inference data exists before initializing servo controller
    if not Action._ensure_inference_data_exists():
        logging.error("Cannot proceed without inference data. Exiting.")
        exit(1)

    servo_controller = ServoController(test_mode=test_mode)
    if not servo_controller.initialize_servos(): 
        exit()

    start_time = time.time()
    try:
        running = True
        while running:
            if test_mode:
                user_input = input("Press Enter to test inference, 'l' to test lookup, 'q' to quit: ")
            else:
                user_input = input("Press Enter to Inference Another Action, 'q' to quit: ")
                
            if user_input.lower() == "q":
                logging.info("User has opted to End the Action Sequence.")
                running = False
            elif user_input.lower() == "l" and test_mode:
                lookup = Action._get_lookup()
                lookup.test_lookup()
            else:
                servo_controller.reset_to_origin()  
                Action.perform_action(servo_controller)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Exiting Action Sequence.")
    except Exception as e:
        logging.error(f"An error occurred during the action sequence: {e}")
    finally:
        servo_controller.cleanup()  
        logging.info("Servo controller cleaned up successfully.")
            
    end_time = time.time()
    logging.info("-" * 100)
    logging.info(f"Completed Mechanicus Action Pipeline in {end_time-start_time} seconds")
    logging.info("-" * 100)