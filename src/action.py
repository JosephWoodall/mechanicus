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
import pyfirmata2

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
    """Class to handle hash to servo angle lookup"""
    
    def __init__(self, dataset_file="inference_data.json"):
        """Initialize lookup tables from dataset file"""
        self.hash_to_servo = {}
        self.hash_to_position = {}
        self.load_lookup_tables(dataset_file)
    
    def load_lookup_tables(self, dataset_file):
        """Load lookup tables from dataset JSON file"""
        try:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            self.hash_to_servo = metadata.get('hash_to_servo_lookup', {})
            self.hash_to_position = metadata.get('hash_to_position_lookup', {})
            
            if not self.hash_to_servo:
                self._create_lookup_from_samples(data)
                
            logging.info(f"Loaded {len(self.hash_to_servo)} hash-to-servo mappings")
            
        except Exception as e:
            logging.error(f"Error loading lookup tables from {dataset_file}: {e}")
            self.hash_to_servo = {}
            self.hash_to_position = {}
    
    def _create_lookup_from_samples(self, data):
        """Create lookup from sample data if not in metadata"""
        for key, sample in data.items():
            if key != 'metadata':
                hash_val = sample['position_hash']
                if hash_val not in self.hash_to_servo:
                    self.hash_to_servo[hash_val] = sample['servo_angles']
                    self.hash_to_position[hash_val] = sample['position']
    
    def get_servo_angles_from_hash(self, position_hash):
        """Convert position hash to servo angles"""
        servo_angles = self.hash_to_servo.get(position_hash, None)
        if servo_angles is None:
            logging.warning(f"No servo angles found for hash: {position_hash}")
            return [0, 0, 0]  
        return servo_angles
    
    def get_position_from_hash(self, position_hash):
        """Convert position hash to 3D coordinates"""
        position = self.hash_to_position.get(position_hash, None)
        if position is None:
            logging.warning(f"No position found for hash: {position_hash}")
            return [0, 0, 0]  
        return position
    
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
            return
        
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
        """Get or create the hash lookup instance"""
        if cls._lookup is None:
            cls._lookup = HashToServoLookup()
        return cls._lookup

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
        inference_data = DataCollector.load_servo_eeg_data("inference_data.json")
        return inference_data

    def __perform_inference():
        response_variable_production = "position_hash" 
        inference_data = Action.__collect_inference_data()
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