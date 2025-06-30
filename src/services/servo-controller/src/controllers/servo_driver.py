import redis
import json
import logging
import time
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="servo_driver.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

if "--test" not in sys.argv and '-t' not in sys.argv:
    try:
        import pyfirmata2
    except ImportError:
        logging.warning(
            "pyfirmata2 not available - hardware mode may not work")


class ServoDriver:
    """ServoDriver class that receives predicted servo angles from Redis and moves servos accordingly"""

    def __init__(self, redis_url="redis://localhost:6379", test_mode=False):
        """Initialize ServoDriver with Redis connection and servo setup"""
        self.redis_client = redis.Redis.from_url(
            redis_url, decode_responses=True)
        self.test_mode = test_mode
        self.board = None
        self.servos = []
        self.current_positions = [0, 0, 0]
        self.input_channel = "predicted_servo_angles"

        if not self.initialize_servos():
            raise Exception("Failed to initialize servos")

    def initialize_servos(self):
        """Initialize servo connections and move to starting positions"""
        try:
            if self.test_mode:
                logging.info("TEST MODE: Simulating servo initialization")
                self.servos = ["simulated_servo_1",
                               "simulated_servo_2", "simulated_servo_3"]
                logging.info(
                    f"TEST MODE: Servos initialized to starting positions: {self.current_positions}")
                return True
            else:
                port = pyfirmata2.Arduino.AUTODETECT
                self.board = pyfirmata2.Arduino(port)

                time.sleep(2)

                servo_pins = ["d:9:s", "d:10:s", "d:11:s"]
                for pin in servo_pins:
                    servo = self.board.get_pin(pin)
                    self.servos.append(servo)

                self.move_to_positions([0, 0, 0])
                logging.info(
                    f"Servos initialized to starting positions: {self.current_positions}")
                return True
        except Exception as e:
            logging.error(f"Failed to initialize servos: {e}")
            return False

    def move_to_positions(self, positions):
        """Move servos to specific positions"""
        if self.test_mode:
            logging.info(
                f"TEST MODE: Simulating servo movement to positions: {positions}")
            for i, position in enumerate(positions):
                if i < len(self.current_positions):
                    self.current_positions[i] = position
            logging.debug(
                f"TEST MODE: Current positions updated to: {self.current_positions}")
            return

        if len(self.servos) >= len(positions):
            for i, position in enumerate(positions):
                if i < len(self.servos):
                    self.servos[i].write(position)
                    self.current_positions[i] = position
            logging.debug(
                f"Servos moved to positions: {self.current_positions}")

    def get_current_position(self):
        """Get current servo positions"""
        return self.current_positions

    def reset_to_origin(self):
        """Reset all servos to position 0"""
        if self.test_mode:
            logging.info("TEST MODE: Simulating reset to origin positions")
            self.current_positions = [0, 0, 0]
            logging.debug(
                f"TEST MODE: Current positions reset to: {self.current_positions}")
            return

        self.move_to_positions([0, 0, 0])
        logging.info(
            f"Servos reset to origin positions: {self.current_positions}")

    def handle_servo_command(self, message):
        """Process servo angle command from Redis channel"""
        try:
            data = json.loads(message['data'])

            servo_angles = data.get('servo_angles', data.get('servo_path', []))

            if not servo_angles:
                logging.warning("No servo angles found in message")
                return

            if isinstance(servo_angles[0], list):
                logging.info(
                    f"Executing servo path with {len(servo_angles)} positions")
                for i, angles in enumerate(servo_angles):
                    logging.info(
                        f"Moving to position {i+1}/{len(servo_angles)}: {angles}")
                    self.move_to_positions(angles)
                    time.sleep(0.5)  
            else:
                logging.info(f"Moving servos to position: {servo_angles}")
                self.move_to_positions(servo_angles)

            timestamp = data.get('timestamp', datetime.now().isoformat())
            target_hash = data.get('target_hash', 'unknown')
            source = data.get('source', 'unknown')

            logging.info(
                f"Servo movement completed - Timestamp: {timestamp}, Hash: {target_hash}, Source: {source}")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON message: {e}")
        except Exception as e:
            logging.error(f"Error processing servo command: {e}")

    def run_servo_listener(self):
        """Main loop to listen for servo commands from Redis"""
        logging.info(
            f"Starting servo driver listener on channel: {self.input_channel}")
        logging.info(
            f"Running in {'TEST' if self.test_mode else 'HARDWARE'} mode")

        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.input_channel)

        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    logging.debug(
                        f"Received message from {self.input_channel}: {message['data']}")
                    self.handle_servo_command(message)

        except KeyboardInterrupt:
            logging.info("Servo driver interrupted by user")
        except Exception as e:
            logging.error(f"Error in servo listener: {e}")
        finally:
            pubsub.unsubscribe(self.input_channel)
            pubsub.close()
            self.cleanup()

    def cleanup(self):
        """Clean up board connection"""
        if self.board and not self.test_mode:
            try:
                self.board.exit()
                logging.info("Board connection cleaned up successfully.")
            except Exception as e:
                logging.error(f"Error cleaning up board connection: {e}")


def main():
    """Main function to run the servo driver service"""
    test_mode = "--test" in sys.argv or '-t' in sys.argv

    logging.info("-" * 100)
    if test_mode:
        logging.info("Starting Servo Driver Service (TEST MODE)")
        print("Running in TEST MODE: Simulating servo movements without hardware.")
    else:
        logging.info("Starting Servo Driver Service")
        print("Running in HARDWARE MODE: Interacting with physical servos.")
    logging.info("-" * 100)

    try:
        servo_driver = ServoDriver(test_mode=test_mode)
        servo_driver.run_servo_listener()
    except Exception as e:
        logging.error(f"Failed to start servo driver: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
