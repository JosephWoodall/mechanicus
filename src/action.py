import logging

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
    
    def __init__(self):
        self.board = None
        self.servo = None
        self.current_position = 0
    
    def initialize_servo(self):
        """Initialize servo connection and move to starting position"""
        try:
            port = pyfirmata2.Arduino.AUTODETECT
            self.board = pyfirmata2.Arduino(port)
            self.servo = self.board.get_pin("d:9:s")
            self.move_to_position(0)
            logging.info(f"Servo initialized to starting position: {self.current_position}")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize servo: {e}")
            return False
    
    def move_to_position(self, position):
        """Move servo to specific position"""
        if self.servo:
            self.servo.write(position)
            self.current_position = position
            logging.debug(f"Servo moved to position: {self.current_position}")
    
    def reset_to_origin(self):
        """Reset servo to position 0"""
        self.move_to_position(0)
        logging.info(f"Servo reset to origin position: {self.current_position}")
    
    def get_current_position(self):
        """Get current servo position"""
        return self.current_position
                

class Action:
    """This class contains functions that leverage the output of the Inference class and move the servo to the appropriate position"""

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
        # collect the inference data from the device or pass generated data for experimentation purposes if none exists
        generated_inference_data = DataCollector.generate_offline_eeg_data(1, 9759)
        return generated_inference_data

    def __perform_inference():
        reponse_variable_production = "activity_type"
        inference_data = Action.__collect_inference_data()
        preprocessed_inference_data = PreprocessData.preprocess_data(
            inference_data, reponse_variable_production, Phase("inference")
        )
        preprocessed_inference_x = preprocessed_inference_data[0]
        loaded_model = Action.__load_model_for_inference_from_file()
        logging.info("Performing inference using loaded model...")
        try:
            predictions = loaded_model.predict(preprocessed_inference_x)
            predictions_dataframe = pandas.DataFrame({"predicitons": predictions})
            logging.info("...Inference using loaded model complete.")
            inferenced_action = predictions_dataframe.iloc[0, 0]
            return inferenced_action
        except Exception as e:
            logging.error(
                f"An error occured when performing inference using loaded model: {e}"
            )
            return pandas.DataFrame({"predictions: null"})

    def __get_inference_value():
        inference_value = Action.__perform_inference()
        logging.info(f"Inferenced Action from Model: {inference_value}")
        return inference_value

    def perform_action(servo_controller: ServoController) -> None:
        """perform_action function is the location in which the machine learning algorithm interacts with hardware.
        """
        logging.info("-" * 100)
        logging.debug(
            "NEW ACTION"
        )
        logging.info("-" * 100)
        try:
            
            logging.debug(f"Current Servo Position: {servo_controller.get_current_position()}")

            inference_value = Action.__get_inference_value()
            logging.info(f"Inferenced Action from Model: {inference_value}")
            
            movement_amount = 90
            
            if inference_value == "baseline_eyes_open":
                logging.debug(f"Move the Servo {movement_amount} degrees (Position 1)")
                new_position = (servo_controller.get_current_position() + movement_amount) 
                servo_controller.move_to_position(new_position)
                
            elif inference_value == "baseline_eyes_closed":
                logging.debug(f"Move the Servo {movement_amount} degrees (Position 2)")
                new_position = (servo_controller.get_current_position() + movement_amount)
                servo_controller.move_to_position(new_position)
            
            logging.info(f"Final Servo Position: {servo_controller.get_current_position()}")
            
        except Exception as e:
            logging.debug(
                f"""Something went wrong when attempting to perform an action: {e}\nExiting action phase."""
            )
            exit()

    def lookup_tuple_value_from_hash(
        inference_data: pandas.DataFrame = None, hash_value: str = None
    ):
        """
        Look up the move_to_position tuple value using a hash value.

        Args:
            inference_data (pd.DataFrame): DataFrame containing 'prosthetic_cartesian_3d_position' and 'prosthetic_cartesian_3d_position_hash_value' columns
            hash_value (int): The prosthetic_cartesian_3d_position_hash_value value to look up

        Returns:
            tuple: The corresponding move_to_position tuple

        Raises:
            ValueError: If no matching hash value is found
        """
        # Find the matching row
        matching_row = inference_data[
            inference_data["prosthetic_cartesian_3d_position_hash_value"] == hash_value
        ]

        # Check if we found exactly one match
        if len(matching_row) == 0:
            raise ValueError(f"No tuple found for hash value {hash_value}")
        elif len(matching_row) > 1:
            raise ValueError(f"Multiple tuples found for hash value {hash_value}")

        # Return the move_to_position tuple
        return matching_row["move_to_position"].iloc[0]

    def move_prosthetic_to_appropriate_position(prosthetic_cartesian_3d_position):
        # ingest the tuple value found from the prediction and move the prosthetic to its position
        # please note, the tuple value from the prediciton must be extracted using the prosthetic_cartesian_3d_position_hash_value value
        # simply lookup the prosthetic_cartesian_3d_position_hash_value's respective tuple value, found from prosthetic_cartesian_3d_position, and pass it for the function
        move_to_position = prosthetic_cartesian_3d_position
        return int(move_to_position)


if __name__ == "__main__":
    logging.info("-" * 100)
    logging.info("Starting Mechanicus Action Pipeline")
    logging.info("-" * 100)
    logging.info("-" * 50)
    logging.debug("This is the Action Sequence. This is where the machine learning model interacts with the hardware.")
    logging.info("-" * 50)

    # Initialize servo controller
    servo_controller = ServoController()
    if not servo_controller.initialize_servo():
        exit()

    start_time = time.time()

    running = True
    while running:
        user_input = input("Press Enter to Inference Another Action, 'q' to quit: ")
        if user_input == "q":
            logging.info("User has opted to End the Action Sequence.")
            running = False
        else:
            servo_controller.reset_to_origin()  # Reset to position 0 before each action
            Action.perform_action(servo_controller)
            
    end_time = time.time()
    logging.info("-" * 100)
    logging.info(f"Completed Mechanicus Action Pipeline in {end_time-start_time} seconds")
    logging.info("-" * 100)
