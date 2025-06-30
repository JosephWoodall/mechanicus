import json
import yaml
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RLTraining:
    def __init__(self, config, training_data_path='/app/shared/data/training_data.json'):
        self.config = config
        self.agent = []
        self.training_data = None
        self.current_position = None

        # Load training data upon initialization
        logger.info("Initializing RL Training - Loading training data...")

        try:
            # Check if file exists
            if not Path(training_data_path).exists():
                logger.error(
                    f"Training data file not found: {training_data_path}")
                raise Exception(
                    f"Training data file not found: {training_data_path}")

            with open(training_data_path, 'r') as file:
                self.training_data = json.load(file)

            logger.info(
                f"Successfully loaded {len(self.training_data)} training samples from {training_data_path}")

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from {training_data_path}: {e}")
            raise Exception(
                f"Failed to decode JSON from {training_data_path}: {e}")
        except Exception as e:
            logger.error(
                f"Failed to load training data from {training_data_path}: {e}")
            raise Exception(
                f"Failed to load training data during initialization: {e}")

        logger.info(
            "RL Training initialized successfully with training data loaded")

    def train(self):
        """Train the RL agent on loaded data"""
        if not self.training_data:
            logger.warning(
                "No training data available. Please load training data first.")
            return

        logger.info(f"Starting training on {len(self.training_data)} samples")

        for i, data in enumerate(self.training_data):
            self.current_position = data['current_position']
            target_position = data['target_position']
            reward = self.calculate_reward(
                self.current_position, target_position)
            # self.agent.update(self.current_position, target_position, reward)
            logging.info(f"Reward: {reward} for sample {i + 1}")
            # Log progress every 100 samples
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(self.training_data)} training samples")

        logger.info("Training completed successfully")

    def calculate_reward(self, current_position, target_position):
        """Calculate reward based on distance between positions"""
        distance = np.linalg.norm(
            np.array(current_position) - np.array(target_position))
        return -distance  # Negative distance as reward

    def save_agent(self, filepath):
        """Save trained agent to file"""
        try:
            pass
            # self.agent.save(filepath)
            # logger.info(f"Agent saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save agent to {filepath}: {e}")


if __name__ == "__main__":
    logger.info("Starting RL Training Pipeline")

    try:
        # Load configuration
        config_path = '/app/shared/config/mechanicus_run_configuration.yaml'
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            exit(1)

        with open(config_path, 'r') as config_file:
            # Changed to yaml.safe_load for YAML files
            config = yaml.safe_load(config_file)

        logger.info(f"Configuration loaded from {config_path}")

        # Initialize trainer - training data will be loaded automatically during initialization
        trainer = RLTraining(config)

        # Train the agent (training data is already loaded)
        # trainer.train()

        # Save the trained agent
        # trainer.save_agent('rl_agent.pkl')

        logger.info("RL Training Pipeline completed successfully")

    except Exception as e:
        logger.error(f"RL Training Pipeline failed: {e}")
        exit(1)
