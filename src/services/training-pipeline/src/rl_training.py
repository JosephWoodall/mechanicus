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
            return self._get_obs(), reward, done, {'distance_to_goal': np.linalg.norm(self.current_pos - self.goal_pos)}

        if self._check_obstacle_collision(new_pos):
            reward = -10.0
            done = True
            return self._get_obs(), reward, done, {'distance_to_goal': np.linalg.norm(self.current_pos - self.goal_pos)}

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

    def clone(self):
        """Create a copy of this agent"""
        new_agent = PPOAgent(self.state_dim, self.action_dim, self.lr)
        new_agent.actor.load_state_dict(self.actor.state_dict())
        new_agent.critic.load_state_dict(self.critic.state_dict())
        new_agent.gamma = self.gamma
        new_agent.clip_epsilon = self.clip_epsilon
        new_agent.k_epochs = self.k_epochs
        return new_agent


class RLAgent:
    """Main RL Agent class for 3D path planning with Tournament Elimination"""

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
                print(
                    f"    Episode {episode}, Average Reward: {avg_reward:.2f}")

        return agent, episode_rewards

    def continue_training_agent(self, agent, start_pos, goal_pos, episodes=500):
        """Continue training an existing agent"""
        obstacles = self.create_sample_obstacles()
        env = Path3DEnvironment(start_pos, goal_pos, obstacles)

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

        return episode_rewards

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

    def evaluate_agent_on_all_pairs(self, agent, num_trials_per_pair=5):
        """Evaluate a single agent on all training pairs"""
        if not self.training_pairs:
            return None

        pair_performances = []
        total_score = 0
        total_success = 0
        total_trials = 0

        for pair_idx, pair in enumerate(self.training_pairs):
            start_pos = np.array(pair['start'])
            goal_pos = np.array(pair['goal'])

            pair_scores = []
            pair_successes = []

            for trial in range(num_trials_per_pair):
                obstacles = self.create_sample_obstacles()
                env = Path3DEnvironment(start_pos, goal_pos, obstacles)

                state = env.reset()
                total_reward = 0
                steps_taken = 0
                done = False

                while not done and steps_taken < 1000:
                    action = agent.get_action(state)
                    state, reward, done, info = env.step(action)
                    total_reward += reward
                    steps_taken += 1

                final_distance = info.get('distance_to_goal', float('inf'))
                success = final_distance < 1.0
                efficiency_bonus = (1000 - steps_taken) * \
                    0.1 if success else -steps_taken * 0.05

                trial_score = total_reward + \
                    (100 if success else 0) + efficiency_bonus
                pair_scores.append(trial_score)
                pair_successes.append(success)

            avg_pair_score = np.mean(pair_scores)
            pair_success_rate = np.mean(pair_successes)

            pair_performances.append({
                'pair_index': pair_idx,
                'start': start_pos.tolist(),
                'goal': goal_pos.tolist(),
                'avg_score': avg_pair_score,
                'success_rate': pair_success_rate,
                'trial_scores': pair_scores
            })

            total_score += avg_pair_score
            total_success += pair_success_rate
            total_trials += 1

        overall_performance = {
            'average_score_across_all_pairs': total_score / total_trials if total_trials > 0 else 0,
            'average_success_rate_across_all_pairs': total_success / total_trials if total_trials > 0 else 0,
            'total_pairs_tested': total_trials,
            'pair_by_pair_performance': pair_performances
        }

        return overall_performance

    def tournament_elimination_training(self, round1_episodes=5, round2_episodes=10, round3_episodes=15, evaluation_trials=5, final_trials=10):
        """Exact implementation of the 4-stage tournament outline with configurable episodes"""
        if not self.training_pairs:
            self.load_training_data()

        if not self.training_pairs:
            print("No training pairs available. Cannot proceed with tournament.")
            return None

        print(f"\nTOURNAMENT ELIMINATION CHAMPIONSHIP")
        print(f"{'='*80}")
        print(f"Initial Contestants: {len(self.training_pairs)} agents")
        print(f"Fixed 4-Stage Tournament Structure")
        print(
            f"Episode Configuration: R1={round1_episodes}, R2={round2_episodes}, R3={round3_episodes}")
        print(
            f"Evaluation Trials: {evaluation_trials}, Final Trials: {final_trials}")
        print(f"{'='*80}")

        print(f"\nROUND 1: MODERATE TRAINING ({round1_episodes} episodes)")
        print(f"{'='*60}")
        current_agents = {}

        for i, pair in enumerate(self.training_pairs):
            print(f"\nTraining Contestant {i+1}/{len(self.training_pairs)}")
            print(f"  Specializing on: {pair['start']} → {pair['goal']}")

            agent, rewards = self.train_agent(
                pair['start'], pair['goal'], episodes=round1_episodes)

            current_agents[i] = {
                'agent': agent,
                'training_pair': pair,
                'training_rewards': rewards,
                'agent_id': i,
                'tournament_round': 1,
                'total_training_episodes': round1_episodes
            }

        print(f"\nROUND 1 EVALUATION ({evaluation_trials} trials per pair)")
        agent_performances = {}
        for agent_id, agent_data in current_agents.items():
            performance = self.evaluate_agent_on_all_pairs(
                agent_data['agent'], num_trials_per_pair=evaluation_trials)
            agent_performances[agent_id] = {
                'performance': performance,
                'score': performance['average_score_across_all_pairs']
            }

        keep_count_r2 = max(2, len(current_agents) // 2)  # Top 50%
        survivors_r2 = sorted(agent_performances.items(
        ), key=lambda x: x[1]['score'], reverse=True)[:keep_count_r2]

        print(f"\nROUND 2: INTENSIVE TRAINING ({round2_episodes} episodes)")
        print(
            f"Advancing {len(survivors_r2)}/{len(current_agents)} agents (Top 50%)")

        enhanced_agents_r2 = {}
        for agent_id, perf_data in survivors_r2:
            agent_data = current_agents[agent_id]
            print(
                f"  Enhancing Agent {agent_id} with {round2_episodes} additional episodes...")

            additional_rewards = self.continue_training_agent(
                agent_data['agent'],
                agent_data['training_pair']['start'],
                agent_data['training_pair']['goal'],
                episodes=round2_episodes
            )

            enhanced_agents_r2[agent_id] = {
                'agent': agent_data['agent'],
                'training_pair': agent_data['training_pair'],
                'training_rewards': agent_data['training_rewards'] + additional_rewards,
                'agent_id': agent_id,
                'tournament_round': 2,
                'total_training_episodes': round1_episodes + round2_episodes,
                'previous_performance': perf_data['performance']
            }

        print(f"\nROUND 2 EVALUATION ({evaluation_trials} trials per pair)")
        agent_performances_r2 = {}
        for agent_id, agent_data in enhanced_agents_r2.items():
            performance = self.evaluate_agent_on_all_pairs(
                agent_data['agent'], num_trials_per_pair=evaluation_trials)
            agent_performances_r2[agent_id] = {
                'performance': performance,
                'score': performance['average_score_across_all_pairs']
            }

        keep_count_r3 = max(2, len(enhanced_agents_r2) //
                            4)
        if keep_count_r3 > len(enhanced_agents_r2) // 2:
            keep_count_r3 = max(2, len(enhanced_agents_r2) // 2)

        survivors_r3 = sorted(agent_performances_r2.items(
        ), key=lambda x: x[1]['score'], reverse=True)[:keep_count_r3]

        print(
            f"\nROUND 3: CHAMPIONSHIP TRAINING ({round3_episodes} episodes)")
        print(
            f"Advancing {len(survivors_r3)}/{len(enhanced_agents_r2)} agents (Elite Championship Level)")

        enhanced_agents_r3 = {}
        for agent_id, perf_data in survivors_r3:
            agent_data = enhanced_agents_r2[agent_id]
            print(
                f"  Championship training for Agent {agent_id} with {round3_episodes} additional episodes...")

            additional_rewards = self.continue_training_agent(
                agent_data['agent'],
                agent_data['training_pair']['start'],
                agent_data['training_pair']['goal'],
                episodes=round3_episodes
            )

            enhanced_agents_r3[agent_id] = {
                'agent': agent_data['agent'],
                'training_pair': agent_data['training_pair'],
                'training_rewards': agent_data['training_rewards'] + additional_rewards,
                'agent_id': agent_id,
                'tournament_round': 3,
                'total_training_episodes': round1_episodes + round2_episodes + round3_episodes,
                'previous_performance': perf_data['performance']
            }

        print(
            f"\nFINAL: ULTIMATE HEAD-TO-HEAD ({final_trials} trials per pair)")
        print(f"{'='*60}")

        final_performances = {}
        for agent_id, agent_data in enhanced_agents_r3.items():
            performance = self.evaluate_agent_on_all_pairs(
                agent_data['agent'], num_trials_per_pair=final_trials)
            final_performances[agent_id] = performance

        final_2 = sorted(final_performances.items(
        ), key=lambda x: x[1]['average_score_across_all_pairs'], reverse=True)[:2]

        if len(final_2) == 2:
            print(f"FINALISTS:")
            for i, (agent_id, performance) in enumerate(final_2, 1):
                agent_data = enhanced_agents_r3[agent_id]
                print(
                    f"  {i}. Agent {agent_id} | Score: {performance['average_score_across_all_pairs']:.2f} | Episodes: {agent_data['total_training_episodes']}")

            champion_id = final_2[0][0]
            champion_performance = final_2[0][1]
        else:
            champion_id = list(enhanced_agents_r3.keys())[0]
            champion_performance = final_performances[champion_id]

        champion_data = enhanced_agents_r3[champion_id]
        champion_data['global_performance'] = champion_performance

        print(f"\n{'='*80}")
        print(f"ULTIMATE TOURNAMENT CHAMPION: Agent {champion_id}")
        print(f"{'='*80}")
        print(
            f"Episode Configuration Used: R1={round1_episodes}, R2={round2_episodes}, R3={round3_episodes}")
        print(
            f"Total Training Episodes: {champion_data['total_training_episodes']}")
        print(
            f"Final Score: {champion_performance['average_score_across_all_pairs']:.2f}")
        print(
            f"Success Rate: {champion_performance['average_success_rate_across_all_pairs']:.2%}")

        self.trained_agents = {champion_id: champion_data}

        return {
            'champion_id': champion_id,
            'champion_data': champion_data,
            'tournament_results': {
                'initial_contestants': len(self.training_pairs),
                'round_1_episodes': round1_episodes,
                'round_2_episodes': round2_episodes,
                'round_3_episodes': round3_episodes,
                'evaluation_trials': evaluation_trials,
                'final_trials': final_trials,
                'round_1_survivors': len(survivors_r2),
                'round_2_survivors': len(survivors_r3),
                'final_2': len(final_2),
                'champion_training_episodes': champion_data['total_training_episodes'],
                'champion_global_score': champion_performance['average_score_across_all_pairs'],
                'champion_success_rate': champion_performance['average_success_rate_across_all_pairs']
            }
        }

    def save_tournament_champion(self, model_path='src/shared/models/rl_tournament_champion.pkl', tournament_results=None):
        """Save the tournament champion"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            if not self.trained_agents:
                print("No tournament champion to save!")
                return False

            champion_id = list(self.trained_agents.keys())[0]
            champion_data = self.trained_agents[champion_id]

            save_data = {
                'training_pairs': self.training_pairs,
                'tournament_champion_id': champion_id,
                'tournament_champion': {
                    'original_training_pair': champion_data['training_pair'],
                    'training_rewards': champion_data['training_rewards'],
                    'tournament_round': champion_data['tournament_round'],
                    'total_training_episodes': champion_data['total_training_episodes'],
                    'global_performance': champion_data.get('global_performance', champion_data.get('previous_performance', {})),
                    'agent_state_dict': {
                        'actor': champion_data['agent'].actor.state_dict(),
                        'critic': champion_data['agent'].critic.state_dict(),
                        'actor_optimizer': champion_data['agent'].actor_optimizer.state_dict(),
                        'critic_optimizer': champion_data['agent'].critic_optimizer.state_dict(),
                        'hyperparams': {
                            'state_dim': champion_data['agent'].state_dim,
                            'action_dim': champion_data['agent'].action_dim,
                            'lr': champion_data['agent'].lr,
                            'gamma': champion_data['agent'].gamma,
                            'clip_epsilon': champion_data['agent'].clip_epsilon,
                            'k_epochs': champion_data['agent'].k_epochs
                        }
                    }
                },
                'model_metadata': {
                    'training_method': 'tournament_elimination',
                    'initial_contestants': tournament_results['tournament_results']['initial_contestants'] if tournament_results else len(self.training_pairs),
                    'elimination_rounds': tournament_results['tournament_results']['elimination_rounds'] if tournament_results else 'unknown',
                    'total_training_episodes': champion_data['total_training_episodes'],
                    'champion_selected': True,
                    'save_timestamp': np.datetime64('now').astype(str)
                }
            }

            if tournament_results:
                save_data['tournament_results'] = tournament_results['tournament_results']

            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)

            print(
                f"\n🏆 Successfully saved TOURNAMENT CHAMPION to: {model_path}")
            print(
                f"Champion survived {champion_data['tournament_round']} tournament rounds")
            print(
                f"Total training episodes: {champion_data['total_training_episodes']}")
            return True

        except Exception as e:
            print(f"Error saving tournament champion: {e}")
            return False

    def load_tournament_champion(self, model_path='/app/shared/models/rl_tournament_champion.pkl'):
        """Load the tournament champion agent"""
        try:
            if not os.path.exists(model_path):
                print(f"Tournament champion model not found: {model_path}")
                return False

            with open(model_path, 'rb') as f:
                save_data = pickle.load(f)

            self.training_pairs = save_data.get('training_pairs', [])

            champion_data = save_data.get('tournament_champion', {})
            champion_id = save_data.get('tournament_champion_id', 0)

            if not champion_data:
                print("No tournament champion data found in model file")
                return False

            hyperparams = champion_data['agent_state_dict']['hyperparams']
            agent = PPOAgent(
                state_dim=hyperparams['state_dim'],
                action_dim=hyperparams['action_dim'],
                lr=hyperparams['lr']
            )

            agent.gamma = hyperparams['gamma']
            agent.clip_epsilon = hyperparams['clip_epsilon']
            agent.k_epochs = hyperparams['k_epochs']

            agent.actor.load_state_dict(
                champion_data['agent_state_dict']['actor'])
            agent.critic.load_state_dict(
                champion_data['agent_state_dict']['critic'])
            agent.actor_optimizer.load_state_dict(
                champion_data['agent_state_dict']['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(
                champion_data['agent_state_dict']['critic_optimizer'])

            self.trained_agents = {champion_id: {
                'agent': agent,
                'training_pair': champion_data['original_training_pair'],
                'training_rewards': champion_data['training_rewards'],
                'tournament_round': champion_data['tournament_round'],
                'total_training_episodes': champion_data['total_training_episodes'],
                'global_performance': champion_data['global_performance']
            }}

            metadata = save_data.get('model_metadata', {})
            tournament_info = save_data.get('tournament_results', {})

            print(
                f"🏆 Successfully loaded TOURNAMENT CHAMPION from: {model_path}")
            print(f"Champion ID: {champion_id}")
            print(
                f"Originally trained on: {champion_data['original_training_pair']['start']} → {champion_data['original_training_pair']['goal']}")
            print(f"Tournament Performance:")
            print(
                f"  • Survived {champion_data['tournament_round']} elimination rounds")
            print(
                f"  • Total training episodes: {champion_data['total_training_episodes']}")

            perf = champion_data['global_performance']
            print(
                f"  • Final Score: {perf['average_score_across_all_pairs']:.2f}")
            print(
                f"  • Final Success Rate: {perf['average_success_rate_across_all_pairs']:.2%}")
            print(f"  • Tested on {perf['total_pairs_tested']} position pairs")
            print(
                f"Selected from {metadata.get('initial_contestants', 'unknown')} initial contestants")

            return True

        except Exception as e:
            print(f"Error loading tournament champion: {e}")
            return False

    def generate_smooth_path(self, agent_id, num_smooth_points=100):
        """Generate and smooth path for the champion agent"""
        if agent_id not in self.trained_agents:
            print(f"No trained agent for ID {agent_id}")
            return None

        agent_data = self.trained_agents[agent_id]
        agent = agent_data['agent']

        if 'start' in agent_data:
            start_pos = np.array(agent_data['start'])
            goal_pos = np.array(agent_data['goal'])
        else:
            start_pos = np.array(agent_data['training_pair']['start'])
            goal_pos = np.array(agent_data['training_pair']['goal'])

        raw_path = self.generate_path(agent, start_pos, goal_pos)
        smooth_path = self.smooth_path(raw_path, num_smooth_points)

        return {
            'raw_path': raw_path,
            'smooth_path': smooth_path,
            'start': start_pos,
            'goal': goal_pos
        }

    def get_model_info(self, model_path='/app/shared/models/rl_tournament_champion.pkl'):
        """Get information about a saved tournament champion model"""
        try:
            if not os.path.exists(model_path):
                return None

            with open(model_path, 'rb') as f:
                save_data = pickle.load(f)

            metadata = save_data.get('model_metadata', {})
            tournament_info = save_data.get('tournament_results', {})
            champion_data = save_data.get('tournament_champion', {})

            return {
                'model_path': model_path,
                'training_method': metadata.get('training_method', 'unknown'),
                'initial_contestants': metadata.get('initial_contestants', 'unknown'),
                'elimination_rounds': metadata.get('elimination_rounds', 'unknown'),
                'total_training_episodes': metadata.get('total_training_episodes', 'unknown'),
                'save_timestamp': metadata.get('save_timestamp', 'unknown'),
                'champion_performance': {
                    'global_score': champion_data.get('global_performance', {}).get('average_score_across_all_pairs', 'unknown'),
                    'success_rate': champion_data.get('global_performance', {}).get('average_success_rate_across_all_pairs', 'unknown')
                }
            }

        except Exception as e:
            print(f"Error reading model info: {e}")
            return None


if __name__ == "__main__":
    rl_agent = RLAgent()

    model_path = '/app/shared/models/rl_tournament_champion.pkl'
    model_info = rl_agent.get_model_info(model_path)

    if model_info:
        print(f"Found existing tournament champion: {model_info}")
        print("Loading existing champion...")
        if rl_agent.load_tournament_champion(model_path):
            print("Successfully loaded existing tournament champion!")
        else:
            print("Failed to load existing champion, will run new tournament...")
    else:
        print("No existing tournament champion found, will run tournament elimination...")

    if not rl_agent.training_pairs:
        training_pairs = rl_agent.load_training_data()
    else:
        training_pairs = rl_agent.training_pairs

    if training_pairs:
        print("Training data loaded successfully!")

        if not rl_agent.trained_agents:
            print(
                f"\nStarting TOURNAMENT ELIMINATION with {len(training_pairs)} initial contestants...")
            print("Only the strongest will survive!")

            tournament_results = rl_agent.tournament_elimination_training()

            if tournament_results:
                print("\n💾 Saving the tournament champion...")
                rl_agent.save_tournament_champion(
                    model_path, tournament_results)
        else:
            print(f"\nUsing existing tournament champion")

        if rl_agent.trained_agents:
            champion_id = list(rl_agent.trained_agents.keys())[0]
            path_data = rl_agent.generate_smooth_path(champion_id)
            if path_data:
                print(
                    f"\nGenerated sample path with {len(path_data['smooth_path'])} smooth points")
                print(
                    f"Path goes from {path_data['start']} to {path_data['goal']}")
    else:
        print("Could not load training data. Please check file location and format.")
