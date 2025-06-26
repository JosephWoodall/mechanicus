# Import necessary libraries
import yaml
import time
import json
from controllers.servo_driver import ServoDriver
from rl_agent.agent import RLAgent

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Initialize the servo controller service
def initialize_service(config):
    servo_driver = ServoDriver(config['servo_config'])
    rl_agent = RLAgent(config['rl_agent_params'])
    return servo_driver, rl_agent

# Main execution flow
def main():
    config = load_config('../shared/config/mechanicus_run_configuration.yaml')
    servo_driver, rl_agent = initialize_service(config)

    # Device Powers On
    print("Device is powering on...")
    time.sleep(1)  # Simulate loading time
    print("Initialization complete.")

    try:
        while True:
            # EEG Monitoring Loop
            # Here you would integrate with the EEG data source
            # For demonstration, we will simulate a position input
            current_position = servo_driver.get_current_position()
            target_position = rl_agent.predict_target_position(current_position)

            # Action Execution
            servo_driver.move_to_position(target_position)
            print(f"Moving from {current_position} to {target_position}")

            time.sleep(1)  # Simulate time taken for movement

    except KeyboardInterrupt:
        print("Shutting down the servo controller service.")

if __name__ == "__main__":
    main()