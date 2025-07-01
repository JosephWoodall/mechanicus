import json
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import CubicSpline
import os
import pickle


class Path3DEnvironment(gym.Env):
    """Custom 3D path planning environment for OpenAI Gym"""

    def __init__(self, start_pos, goal_pos, obstacles=None, max_steps=1000):
        super(Path3DEnvironment, self).__init__()

        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.current_pos = self.start_pos.copy()
        self.obstacles = obstacles or []
        self.max_steps = max_steps
        self.step_count = 0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(6,), dtype=np.float32
        )

        self.bounds = {
            'x': (-50, 50),
            'y': (-50, 50),
            'z': (-50, 50)
        }

    def reset(self):
        """Reset environment to initial state"""
        self.current_pos = self.start_pos.copy()
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        """Get current observation"""
        return np.concatenate([self.current_pos, self.goal_pos])

    def _is_in_bounds(self, pos):
        """Check if position is within environment bounds"""
        return (self.bounds['x'][0] <= pos[0] <= self.bounds['x'][1] and
                self.bounds['y'][0] <= pos[1] <= self.bounds['y'][1] and
                self.bounds['z'][0] <= pos[2] <= self.bounds['z'][1])

    def _check_obstacle_collision(self, pos):
        """Check if position collides with obstacles"""
        for obstacle in self.obstacles:
            center, radius = obstacle['center'], obstacle['radius']
            if np.linalg.norm(pos - np.array(center)) <= radius:
                return True
        return False

    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        new_pos = self.current_pos + action * 2.0  

        if not self._is_in_bounds(new_pos):
            reward = -10.0  
            done = True
            return self._get_obs(), reward, done, {}

        if self._check_obstacle_collision(new_pos):
            reward = -10.0  
            done = True
            return self._get_obs(), reward, done, {}

        self.current_pos = new_pos

        distance_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        reward = -distance_to_goal  

        done = distance_to_goal < 1.0 or self.step_count >= self.max_steps

        if distance_to_goal < 1.0:
            reward += 100.0  

        return self._get_obs(), reward, done, {'distance_to_goal': distance_to_goal}


class PPOAgent:
    """Proximal Policy Optimization Agent"""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.k_epochs = 4

    def _build_actor(self):
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()  
        )

    def _build_critic(self):
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_action(self, state):
        """Get action from current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        """Update PPO agent"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()

        returns = rewards + self.gamma * next_values * ~dones
        advantages = returns - values

        for _ in range(self.k_epochs):
            current_actions = self.actor(states)
            action_loss = nn.MSELoss()(current_actions, actions)

            current_values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(current_values, returns.detach())

            self.actor_optimizer.zero_grad()
            action_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


class RLAgent:
    """Main RL Agent class for 3D path planning"""

    def __init__(self, data_file=None):
        possible_paths = [
            data_file,
            '/app/shared/data/training_data.json',
            'training_data.json',
            '../training_data.json',
            '../../training_data.json',
            '../../../training_data.json',
            os.path.join(os.path.dirname(__file__), 'training_data.json'),
            os.path.join(os.path.dirname(__file__),
                         '..', 'training_data.json'),
            os.path.join(os.path.dirname(__file__), '..',
                         '..', 'training_data.json')
        ]

        self.data_file = None
        for path in possible_paths:
            if path and os.path.exists(path):
                self.data_file = path
                print(f"Found training data at: {path}")
                break

        if not self.data_file:
            print("Training data file not found. Checked locations:")
            for path in possible_paths:
                if path:
                    print(f"  - {path}")

        self.training_pairs = []
        self.trained_agents = {}

    def create_sample_data(self):
        """Create sample training data matching the expected format"""
        sample_data = {
            "dataset_metadata": {
                "total_samples": 5,
                "anomaly_samples": 0,
                "normal_samples": 5,
                "anomaly_rate": 0.0,
                "n_channels": 5,
                "n_servos": 3,
                "total_positions": 5,
                "training_samples": 5,
                "generated_timestamp": "2025-07-01T16:33:42.900742",
                "source": "MLTrainingDatasetGenerator",
                "dataset_type": "training"
            },
            "training_samples": [
                {
                    "timestamp": "2025-07-01T16:33:42.871156",
                    "sample_id": "training_sample_000001",
                    "eeg_data": [0.18810283812846107, 1.0860538009985155, 1.4487606442695369, 0.05043290974262393, 0.4194449421066657],
                    "n_channels": 5,
                    "is_anomaly": False,
                    "servo_angles": [72.0, 36.0, 108.0],
                    "position": [0.10898137920080414, 0.33541019662496846, 0.4854101966249684],
                    "source": "MLTrainingDatasetGenerator"
                },
                {
                    "timestamp": "2025-07-01T16:33:42.872156",
                    "sample_id": "training_sample_000002",
                    "eeg_data": [0.28810283812846107, 1.1860538009985155, 1.5487606442695369, 0.15043290974262393, 0.5194449421066657],
                    "n_channels": 5,
                    "is_anomaly": False,
                    "servo_angles": [80.0, 45.0, 120.0],
                    "position": [0.20898137920080414, 0.43541019662496846, 0.5854101966249684],
                    "source": "MLTrainingDatasetGenerator"
                },
                {
                    "timestamp": "2025-07-01T16:33:42.873156",
                    "sample_id": "training_sample_000003",
                    "eeg_data": [0.38810283812846107, 1.2860538009985155, 1.6487606442695369, 0.25043290974262393, 0.6194449421066657],
                    "n_channels": 5,
                    "is_anomaly": False,
                    "servo_angles": [90.0, 60.0, 135.0],
                    "position": [0.30898137920080414, 0.53541019662496846, 0.6854101966249684],
                    "source": "MLTrainingDatasetGenerator"
                },
                {
                    "timestamp": "2025-07-01T16:33:42.874156",
                    "sample_id": "training_sample_000004",
                    "eeg_data": [0.48810283812846107, 1.3860538009985155, 1.7487606442695369, 0.35043290974262393, 0.7194449421066657],
                    "n_channels": 5,
                    "is_anomaly": False,
                    "servo_angles": [100.0, 75.0, 150.0],
                    "position": [0.40898137920080414, 0.63541019662496846, 0.7854101966249684],
                    "source": "MLTrainingDatasetGenerator"
                },
                {
                    "timestamp": "2025-07-01T16:33:42.875156",
                    "sample_id": "training_sample_000005",
                    "eeg_data": [0.58810283812846107, 1.4860538009985155, 1.8487606442695369, 0.45043290974262393, 0.8194449421066657],
                    "n_channels": 5,
                    "is_anomaly": False,
                    "servo_angles": [110.0, 90.0, 165.0],
                    "position": [0.50898137920080414, 0.73541019662496846, 0.8854101966249684],
                    "source": "MLTrainingDatasetGenerator"
                }
            ]
        }

        sample_file = 'training_data.json'
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)

        print(f"Created sample training data: {sample_file}")
        self.data_file = sample_file
        return sample_file

    def load_training_data(self):
        """Load and prepare training data from JSON file"""
        if not self.data_file:
            print("No training data file found. Creating sample data...")
            self.create_sample_data()

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            training_samples = data.get('training_samples', [])

            if not training_samples:
                print("No training_samples found in data file")
                return []

            print(f"Found {len(training_samples)} training samples")

            for i in range(len(training_samples)):
                current_sample = training_samples[i]

                if 'servo_angles' not in current_sample:
                    print(f"Warning: servo_angles not found in sample {i}")
                    continue

                servo_angles = current_sample['servo_angles']

                current_pos = servo_angles[:3] if len(
                    servo_angles) >= 3 else servo_angles

                while len(current_pos) < 3:
                    current_pos.append(0.0)

                if i < len(training_samples) - 1:
                    next_sample = training_samples[i + 1]
                    if 'servo_angles' in next_sample:
                        next_servo_angles = next_sample['servo_angles']
                        end_pos = next_servo_angles[:3] if len(
                            next_servo_angles) >= 3 else next_servo_angles
                        while len(end_pos) < 3:
                            end_pos.append(0.0)
                    else:
                        end_pos = [0, 0, 0]
                else:
                    end_pos = [0, 0, 0]

                self.training_pairs.append({
                    'start': current_pos,
                    'goal': end_pos,
                    'sample_id': current_sample.get('sample_id', f'sample_{i}'),
                    'timestamp': current_sample.get('timestamp', 'unknown')
                })

            print(
                f"Loaded {len(self.training_pairs)} training pairs from {self.data_file}")

            print("First few training pairs:")
            for i, pair in enumerate(self.training_pairs[:3]):
                print(
                    f"  Pair {i}: {pair['start']} -> {pair['goal']} (ID: {pair['sample_id']})")

            return self.training_pairs

        except FileNotFoundError:
            print(f"Training data file {self.data_file} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            return []
        except Exception as e:
            print(f"Error loading training data: {e}")
            return []

    def create_sample_obstacles(self):
        """Create sample obstacles for training"""
        obstacles = [
            {'center': [10, 10, 10], 'radius': 5},
            {'center': [-15, 5, -10], 'radius': 7},
            {'center': [0, -20, 15], 'radius': 6}
        ]
        return obstacles

    def train_agent(self, start_pos, goal_pos, episodes=1000):
        """Train PPO agent for specific start-goal pair"""
        obstacles = self.create_sample_obstacles()
        env = Path3DEnvironment(start_pos, goal_pos, obstacles)

        agent = PPOAgent(state_dim=6, action_dim=3)

        episode_rewards = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            states, actions, rewards, next_states, dones = [], [], [], [], []

            while not done:
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_reward += reward

            if len(states) > 0:
                agent.update(states, actions, rewards, next_states, dones)

            episode_rewards.append(episode_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

        return agent, episode_rewards

    def generate_path(self, agent, start_pos, goal_pos, max_steps=1000):
        """Generate path using trained agent"""
        obstacles = self.create_sample_obstacles()
        env = Path3DEnvironment(start_pos, goal_pos, obstacles)

        state = env.reset()
        path = [start_pos.copy()]

        for _ in range(max_steps):
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            path.append(env.current_pos.copy())

            if done:
                break

        return np.array(path)

    def smooth_path(self, path, num_points=100):
        """Smooth path using cubic spline interpolation"""
        if len(path) < 4:
            return path

        t = np.linspace(0, 1, len(path))
        t_smooth = np.linspace(0, 1, num_points)

        cs_x = CubicSpline(t, path[:, 0])
        cs_y = CubicSpline(t, path[:, 1])
        cs_z = CubicSpline(t, path[:, 2])

        smooth_path = np.column_stack([
            cs_x(t_smooth),
            cs_y(t_smooth),
            cs_z(t_smooth)
        ])

        return smooth_path

    def save_agent(self, model_path='src/shared/models/rl_agent.pkl'):
        """Save the trained RL agent to a pickle file"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            save_data = {
                'training_pairs': self.training_pairs,
                'trained_agents': {},
                'model_metadata': {
                    'num_agents': len(self.trained_agents),
                    'training_complete': True,
                    'save_timestamp': np.datetime64('now').astype(str)
                }
            }

            for i, agent_data in self.trained_agents.items():
                save_data['trained_agents'][i] = {
                    'start': agent_data['start'],
                    'goal': agent_data['goal'],
                    'rewards': agent_data['rewards'],
                    'agent_state_dict': {
                        'actor': agent_data['agent'].actor.state_dict(),
                        'critic': agent_data['agent'].critic.state_dict(),
                        'actor_optimizer': agent_data['agent'].actor_optimizer.state_dict(),
                        'critic_optimizer': agent_data['agent'].critic_optimizer.state_dict(),
                        'hyperparams': {
                            'state_dim': agent_data['agent'].state_dim,
                            'action_dim': agent_data['agent'].action_dim,
                            'lr': agent_data['agent'].lr,
                            'gamma': agent_data['agent'].gamma,
                            'clip_epsilon': agent_data['agent'].clip_epsilon,
                            'k_epochs': agent_data['agent'].k_epochs
                        }
                    }
                }

            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"Successfully saved RL agent to: {model_path}")
            print(f"Saved {len(self.trained_agents)} trained agents")
            return True

        except Exception as e:
            print(f"Error saving RL agent: {e}")
            return False

    def load_agent(self, model_path='/app/shared/models/rl_agent.pkl'):
        """Load a trained RL agent from a pickle file"""
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False

            with open(model_path, 'rb') as f:
                save_data = pickle.load(f)

            self.training_pairs = save_data.get('training_pairs', [])

            self.trained_agents = {}
            saved_agents = save_data.get('trained_agents', {})

            for i, agent_data in saved_agents.items():
                hyperparams = agent_data['agent_state_dict']['hyperparams']
                agent = PPOAgent(
                    state_dim=hyperparams['state_dim'],
                    action_dim=hyperparams['action_dim'],
                    lr=hyperparams['lr']
                )

                agent.gamma = hyperparams['gamma']
                agent.clip_epsilon = hyperparams['clip_epsilon']
                agent.k_epochs = hyperparams['k_epochs']

                agent.actor.load_state_dict(
                    agent_data['agent_state_dict']['actor'])
                agent.critic.load_state_dict(
                    agent_data['agent_state_dict']['critic'])
                agent.actor_optimizer.load_state_dict(
                    agent_data['agent_state_dict']['actor_optimizer'])
                agent.critic_optimizer.load_state_dict(
                    agent_data['agent_state_dict']['critic_optimizer'])

                self.trained_agents[int(i)] = {
                    'agent': agent,
                    'start': agent_data['start'],
                    'goal': agent_data['goal'],
                    'rewards': agent_data['rewards']
                }

            metadata = save_data.get('model_metadata', {})
            print(f"Successfully loaded RL agent from: {model_path}")
            print(f"Loaded {len(self.trained_agents)} trained agents")
            print(
                f"Model saved at: {metadata.get('save_timestamp', 'unknown')}")
            return True

        except Exception as e:
            print(f"Error loading RL agent: {e}")
            return False

    def train_all_pairs(self, episodes_per_pair=500, save_after_training=True):
        """Train agents for all start-goal pairs"""
        if not self.training_pairs:
            self.load_training_data()

        if not self.training_pairs:
            print("No training pairs available. Cannot proceed with training.")
            return

        for i, pair in enumerate(self.training_pairs):
            print(f"\nTraining pair {i+1}/{len(self.training_pairs)}")
            print(f"Start: {pair['start']}, Goal: {pair['goal']}")

            agent, rewards = self.train_agent(
                pair['start'],
                pair['goal'],
                episodes=episodes_per_pair
            )

            self.trained_agents[i] = {
                'agent': agent,
                'start': pair['start'],
                'goal': pair['goal'],
                'rewards': rewards
            }

        if save_after_training:
            print("\nSaving trained agents...")
            self.save_agent()

    def get_model_info(self, model_path='/app/shared/models/rl_agent.pkl'):
        """Get information about a saved model without fully loading it"""
        try:
            if not os.path.exists(model_path):
                return None

            with open(model_path, 'rb') as f:
                save_data = pickle.load(f)

            metadata = save_data.get('model_metadata', {})
            training_pairs = save_data.get('training_pairs', [])
            trained_agents = save_data.get('trained_agents', {})

            return {
                'model_path': model_path,
                'num_training_pairs': len(training_pairs),
                'num_trained_agents': len(trained_agents),
                'save_timestamp': metadata.get('save_timestamp', 'unknown'),
                'training_complete': metadata.get('training_complete', False)
            }

        except Exception as e:
            print(f"Error reading model info: {e}")
            return None

    def generate_smooth_path(self, pair_index, num_smooth_points=100):
        """Generate and smooth path for a specific training pair"""
        if pair_index not in self.trained_agents:
            print(f"No trained agent for pair {pair_index}")
            return None

        agent_data = self.trained_agents[pair_index]
        agent = agent_data['agent']
        start_pos = np.array(agent_data['start'])
        goal_pos = np.array(agent_data['goal'])

        raw_path = self.generate_path(agent, start_pos, goal_pos)

        smooth_path = self.smooth_path(raw_path, num_smooth_points)

        return {
            'raw_path': raw_path,
            'smooth_path': smooth_path,
            'start': start_pos,
            'goal': goal_pos
        }


if __name__ == "__main__":
    rl_agent = RLAgent()

    model_path = '/app/shared/models/rl_agent.pkl'
    model_info = rl_agent.get_model_info(model_path)

    if model_info:
        print(f"Found existing model: {model_info}")
        print("Loading existing model...")
        if rl_agent.load_agent(model_path):
            print("Successfully loaded existing model!")
        else:
            print("Failed to load existing model, will train new one...")
    else:
        print("No existing model found, will train new one...")

    if not rl_agent.training_pairs:
        training_pairs = rl_agent.load_training_data()
    else:
        training_pairs = rl_agent.training_pairs

    if training_pairs:
        print("Training data loaded successfully!")
        print(f"First few pairs:")
        for i, pair in enumerate(training_pairs[:3]):
            print(f"  Pair {i}: {pair['start']} -> {pair['goal']}")

        if not rl_agent.trained_agents:
            print("\nStarting training for all pairs...")
            rl_agent.train_all_pairs(
                episodes_per_pair=1, save_after_training=True)
        else:
            print(
                f"\nUsing existing {len(rl_agent.trained_agents)} trained agents")

        if rl_agent.trained_agents:
            path_data = rl_agent.generate_smooth_path(0)
            if path_data:
                print(
                    f"Generated path with {len(path_data['smooth_path'])} smooth points")

                rl_agent.save_agent(model_path)
    else:
        print("Could not load training data. Please check file location and format.")
