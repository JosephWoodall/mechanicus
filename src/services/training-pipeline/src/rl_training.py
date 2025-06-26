from rl_agent.agent import RLAgent
import json
import numpy as np

class RLTraining:
    def __init__(self, config):
        self.config = config
        self.agent = RLAgent(config['rl_agent'])
        self.training_data = []
        self.current_position = None

    def load_training_data(self, filepath):
        with open(filepath, 'r') as file:
            self.training_data = json.load(file)

    def train(self):
        for data in self.training_data:
            self.current_position = data['current_position']
            target_position = data['target_position']
            reward = self.calculate_reward(self.current_position, target_position)
            self.agent.update(self.current_position, target_position, reward)

    def calculate_reward(self, current_position, target_position):
        distance = np.linalg.norm(np.array(current_position) - np.array(target_position))
        return -distance  # Negative distance as reward

    def save_agent(self, filepath):
        self.agent.save(filepath)

if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    trainer = RLTraining(config)
    trainer.load_training_data('path/to/training_data.json')
    trainer.train()
    trainer.save_agent('path/to/rl_agent.pkl')