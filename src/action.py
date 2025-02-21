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
import numpy
import time


from train import DataCollector, PreprocessData, Phase


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
        return inference_value

    def log_inference_value():
        logging.info(f"Inferenced Action from Model: {Action.__get_inference_value()}")

    def perform_action() -> None:
        """perform_action function is the location in which the machine learning algorithm interacts with hardware. Currently, I am obtaining the hardware
        which will interact with the software. I'm thinking of using pyusb to turn a light on/off for baseline_eyes_open/baseline_eyes_closed, just as a poc.
        """
        logging.info("-" * 100)
        logging.debug(
            "YOU WILL PERFORM YOUR ACTIONS HERE. PLEASE SEE BELOW LOGS FOR ACTION LIST."
        )
        logging.info("-" * 100)
        inference_value = Action.__get_inference_value()
        if inference_value == "baseline_eyes_closed":
            # perform some action, like move to this position, turn the light on/off, etc... I'll use a light that is plugged into my pc as an example
            logging.debug(
                "Imagine there is a light in front of you, and you just turned it off by closing your eyes"
            )
        elif inference_value == "baseline_eyes_open":
            logging.debug(
                "Imagine there is a light in front of you, and you just turned it on by opening your eyes"
            )

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
    logging.info("----------STARTING Mechanicus Action Pipeline----------")
    start_time = time.time()

    Action.log_inference_value()
    action = Action.perform_action()
    end_time = time.time()

    logging.info(
        f"----------COMPLETED Mechanicus Action Pipeline in {end_time-start_time} seconds---------"
    )
