import redis
import json
import time
import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServoDriver:
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        """
        Initialize servo driver

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
        """
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("ServoDriver initialized and connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing ServoDriver: {e}")
            raise

    def get_servo_commands(self) -> Optional[Dict[str, Any]]:
        """Get servo commands from Redis"""
        try:
            data = self.redis_client.get("servo_commands")
            if not data:
                return None

            # Parse JSON data
            try:
                command_data = json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON data: {e}")
                logger.error(f"Raw data: {data}")
                return None

            # Validate data structure
            if not isinstance(command_data, dict):
                logger.error(f"Expected dict, got {type(command_data)}")
                return None

            if "servo_commands" not in command_data:
                logger.error("Missing 'servo_commands' key in data")
                return None

            return command_data

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading servo commands: {e}")
            return None

    def validate_servo_commands(self, commands: Any) -> Optional[List[float]]:
        """
        Validate and normalize servo commands

        Args:
            commands: Raw servo commands from Redis

        Returns:
            List of validated servo commands or None if invalid
        """
        try:
            # Handle different input types
            if commands is None:
                logger.warning("Received None servo commands")
                return None

            # Convert to list if needed
            if isinstance(commands, (int, float)):
                commands = [commands]
            elif isinstance(commands, str):
                logger.error(f"Received string servo commands: {commands}")
                return None
            elif not isinstance(commands, (list, tuple)):
                logger.error(f"Invalid servo commands type: {type(commands)}")
                return None

            # Convert to float and validate
            validated_commands = []
            for i, cmd in enumerate(commands):
                try:
                    float_cmd = float(cmd)
                    # Add bounds checking if needed
                    # if not (-180 <= float_cmd <= 180):  # Example bounds
                    #     logger.warning(f"Servo command {i} out of bounds: {float_cmd}")
                    validated_commands.append(float_cmd)
                except (ValueError, TypeError) as e:
                    logger.error(
                        f"Invalid servo command at index {i}: {cmd}, error: {e}")
                    return None

            if not validated_commands:
                logger.warning("No valid servo commands found")
                return None

            return validated_commands

        except Exception as e:
            logger.error(f"Error validating servo commands: {e}")
            return None

    def execute_servo_commands(self, commands: List[float]):
        """
        Execute servo commands

        Args:
            commands: List of validated servo positions/commands
        """
        try:
            # TODO: Implement actual servo control here
            # This is where you'd interface with your servo hardware
            logger.info(
                f"Executing {len(commands)} servo commands: {commands}")

            # Placeholder for actual servo control
            # Example implementations:
            # self.servo_controller.move_servos(commands)
            # for i, cmd in enumerate(commands):
            #     self.servo_controller.set_servo_position(i, cmd)

            # Simulate execution time
            time.sleep(0.01)  # Small delay to simulate servo movement

        except Exception as e:
            logger.error(f"Error executing servo commands: {e}")

    def check_command_freshness(self, timestamp: float, max_age: float = 1.0) -> bool:
        """
        Check if command timestamp is recent enough

        Args:
            timestamp: Command timestamp
            max_age: Maximum age in seconds

        Returns:
            True if command is fresh enough
        """
        try:
            current_time = time.time()
            age = current_time - timestamp

            if age > max_age:
                logger.warning(f"Command too old: {age:.2f}s > {max_age}s")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking command freshness: {e}")
            return False

    def run(self, loop_delay: float = 0.1):
        """
        Main execution loop

        Args:
            loop_delay: Delay between iterations in seconds
        """
        logger.info("Starting servo driver...")

        while True:
            try:
                # Get servo commands from Redis
                command_data = self.get_servo_commands()

                if command_data:
                    # Extract servo commands
                    raw_commands = command_data.get("servo_commands")
                    timestamp = command_data.get("timestamp")

                    # Check command freshness if timestamp provided
                    if timestamp and not self.check_command_freshness(timestamp):
                        logger.warning("Ignoring stale command")
                        continue

                    # Validate servo commands
                    servo_commands = self.validate_servo_commands(raw_commands)

                    if servo_commands:
                        self.execute_servo_commands(servo_commands)
                    else:
                        logger.warning("No valid servo commands to execute")

                time.sleep(loop_delay)

            except KeyboardInterrupt:
                logger.info("Shutting down servo driver...")
                break
            except Exception as e:
                logger.error(f"Error in servo driver loop: {e}")
                time.sleep(loop_delay)


def main():
    """Main entry point"""
    try:
        driver = ServoDriver()
        driver.run()
    except Exception as e:
        logger.error(f"Failed to start servo driver: {e}")


if __name__ == "__main__":
    main()
