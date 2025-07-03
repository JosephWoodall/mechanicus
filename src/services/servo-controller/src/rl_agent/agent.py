import os
import redis
import json
import numpy
import time
import logging
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym
from gymnasium import spaces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/app/shared/models/rl_agent_sb3.zip"


class ServoEnv(gym.Env):
    """
    Custom Environment for the servo controller, compatible with Stable Baselines3.
    """

    def __init__(self):
        super(ServoEnv, self).__init__()
        '''
        State: [current_x, current_y, current_z, end_x, end_y, end_z]; this will be true for 3 servo positions
        If more are needed, then extend the state space accordingly.
        The first three values represent the current servo positions, and the last three are the target positions
        '''
        self.observation_space = spaces.Box(
            low=-numpy.inf, high=numpy.inf, shape=(6,), dtype=numpy.float32)
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(3,), dtype=numpy.float32)
        self.state = numpy.zeros(6)
        self.goal = numpy.zeros(3)
        self.prev_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = numpy.random.uniform(
            0, 180, size=6)  
        self.goal = self.state[3:]
        self.prev_distance = numpy.linalg.norm(self.state[:3] - self.goal)
        return self.state.astype(numpy.float32), {}

    def step(self, action):
        new_pos = numpy.clip(self.state[:3] + action, 0, 180)
        self.state[:3] = new_pos
        distance = numpy.linalg.norm(new_pos - self.state[3:])

        reward = -distance - 0.1 * numpy.linalg.norm(action)
        done = distance < 5.0  
        truncated = False
        self.prev_distance = distance
        return self.state.astype(numpy.float32), reward, done, truncated, {}

    def set_state(self, state):
        self.state = state.copy()
        self.goal = state[3:]
        self.prev_distance = numpy.linalg.norm(state[:3] - self.goal)


def load_rl_agent(env, model_path=MODEL_PATH):
    if os.path.exists(model_path):
        try:
            agent = DDPG.load(model_path, env=env)
            logger.info(f"Successfully loaded RL agent from {model_path}")
            return agent
        except Exception as e:
            logger.warning(
                f"Failed to load agent, will create new instance: {e}")

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=numpy.zeros(
        n_actions), sigma=2.0 * numpy.ones(n_actions))
    agent = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=0,
        tensorboard_log="/app/shared/tb_logs/",
        device="cpu",
        buffer_size=10000,
        learning_starts=100
    )
    return agent


def main():
    logger.info("Starting main function...")

    logger.info("Creating ServoEnv environment...")
    env = DummyVecEnv([lambda: ServoEnv()])
    logger.info("Environment created successfully")

    logger.info("Loading RL agent...")
    rl_agent = load_rl_agent(env)
    logger.info("RL agent loaded successfully")

    logger.info("Connecting to Redis...")
    try:
        redis_client = redis.Redis(
            host='redis', port=6379, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return

    logger.info("=" * 50)
    logger.info("Servo Path Smoothing RL Agent Initialized:")
    logger.info("    Input Channel: predicted_servo_angles")
    logger.info("    Output Key: servo_commands")
    logger.info("    Model Path: " + MODEL_PATH)
    logger.info("    Purpose: Smooth servo movement paths")
    logger.info("=" * 50)

    pubsub = redis_client.pubsub()
    pubsub.subscribe('predicted_servo_angles')
    logger.info("Subscribed to Redis channel: predicted_servo_angles")

    current_servo_positions = [90.0, 90.0, 90.0]  
    last_state = None
    last_action = None
    step_count = 0

    logger.info("Starting servo path smoothing with RL agent...")

    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                step_count += 1
                if step_count % 100 == 0:
                    logger.info(f"Processed {step_count} servo commands")

                try:
                    raw_data = json.loads(message['data'])

                    if 'servo_angles' in raw_data and isinstance(raw_data['servo_angles'], list) and len(raw_data['servo_angles']) > 0:
                        servo_angles_str = raw_data['servo_angles'][0]
                        target_positions = json.loads(servo_angles_str)

                        if len(target_positions) == 3:
                            obs = numpy.array(
                                current_servo_positions + target_positions, dtype=numpy.float32)

                            env.envs[0].set_state(obs)

                            action, _ = rl_agent.predict(
                                obs, deterministic=True)

                            new_positions = numpy.clip(
                                numpy.array(current_servo_positions) + action,
                                0, 180
                            ).tolist()

                            current_servo_positions = new_positions

                            command_data = {
                                "servo_commands": new_positions,
                                "timestamp": time.time(),
                                "target_positions": target_positions,
                                "adjustments": action.tolist()
                            }

                            redis_client.set("servo_commands",
                                             json.dumps(command_data))

                            distance_to_target = numpy.linalg.norm(
                                numpy.array(new_positions) -
                                numpy.array(target_positions)
                            )

                            logger.info(
                                f"Smooth servo path: {new_positions} (target: {target_positions}, distance: {distance_to_target:.1f}°)")

                            if last_state is not None and rl_agent.num_timesteps > rl_agent.learning_starts:
                                try:
                                    distance = numpy.linalg.norm(numpy.array(
                                        new_positions) - numpy.array(target_positions))
                                    movement_smoothness = numpy.linalg.norm(
                                        action)
                                    reward = -distance - 0.1 * movement_smoothness
                                    done = distance < 5.0

                                    rl_agent.replay_buffer.add(
                                        last_state.reshape(1, -1),
                                        last_action.reshape(1, -1),
                                        numpy.array([reward]),
                                        obs.reshape(1, -1),
                                        numpy.array([done]),
                                        numpy.array([[1.0]])
                                    )

                                    if rl_agent.replay_buffer.size() > 64:
                                        rl_agent.train(
                                            batch_size=32, gradient_steps=1)

                                    if done:
                                        logger.info(
                                            "Target reached! Resetting episode.")
                                        last_state, last_action = None, None
                                    else:
                                        last_state = obs
                                        last_action = action

                                except Exception as train_error:
                                    logger.debug(
                                        f"Training error (continuing): {train_error}")
                                    last_state = obs
                                    last_action = action
                            else:
                                last_state = obs
                                last_action = action

                            if step_count % 1000 == 0:
                                try:
                                    rl_agent.save(MODEL_PATH)
                                    logger.info(
                                        f"Saved model at step {step_count}")
                                except Exception as save_error:
                                    logger.debug(
                                        f"Cannot save model (read-only filesystem): {save_error}")

                        else:
                            logger.warning(
                                f"Expected 3 servo angles, got {len(target_positions)}")
                    else:
                        logger.warning("Invalid servo_angles format in data")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        try:
            rl_agent.save(MODEL_PATH)
            logger.info("Final model save completed")
        except:
            logger.info("Could not save final model")
    finally:
        pubsub.unsubscribe('predicted_servo_angles')
        pubsub.close()


if __name__ == "__main__":
    main()
