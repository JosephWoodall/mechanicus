import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="mechanicus_main.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore")

from subprocess import call
import os
from pathlib import Path


def main():
    try:

        logging.debug(f"Current working directory: {os.getcwd()}")
        logging.debug(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")

        try:
            ml_model = Path("inference_model.pkl")
            if not ml_model.exists():
                try:
                    logging.info(
                        "No Loaded Machine Learning Model Exists.Running Training Process Now and Creating Inference Model..."
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
        logging.info("Running Action Sequence Now...")
        try:
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
