import pickle
import redis
import json
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_rl_agent(model_path: str = "/app/shared/models/rl_agent.pkl"):  # Changed path
    """Load the trained RL agent from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
        logger.info(f"Successfully loaded RL agent from {model_path}")
        return agent
    except Exception as e:
        logger.error(f"Error loading RL agent: {e}")
        raise


def main():
    # Load the trained RL agent
    rl_agent = load_rl_agent()

    # Connect to Redis - CHANGED FROM localhost to redis
    redis_client = redis.Redis(
        host='redis', port=6379, decode_responses=True)

    logger.info("Starting servo control with RL agent...")

    while True:
        try:
            # Read data from predicted_servo_angles Redis channel
            data = redis_client.get("predicted_servo_angles")

            if data:
                angle_data = json.loads(data)
                current_position = angle_data.get("current_position", [])
                end_position = angle_data.get("end_position", [])

                if current_position and end_position:
                    # Pass current and end position to the RL agent
                    state = np.array(current_position + end_position)
                    # or whatever method your agent uses
                    servo_commands = rl_agent.predict(state)

                    # Send output to Redis channel for servo_driver.py
                    command_data = {
                        "servo_commands": servo_commands.tolist() if isinstance(servo_commands, np.ndarray) else servo_commands,
                        "timestamp": time.time()
                    }

                    redis_client.set("servo_commands",
                                     json.dumps(command_data))
                    logger.info(
                        f"Sent servo commands: {command_data['servo_commands']}")

            time.sleep(0.1)  # 10Hz update rate

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()
