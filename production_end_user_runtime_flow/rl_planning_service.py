import redis
import json
import logging
import pickle
from datetime import datetime

class RLPlanningService:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.input_channel = "ml_predictions"
        self.output_channel = "servo_commands"
        
        self.rl_agent = self.load_rl_agent()
        self.hash_lookup = self.load_hash_lookup()
        self.current_position = [0, 0, 0]  # Initialize
        
    def load_rl_agent(self):
        """Load trained RL agent"""
        try:
            with open("rl_agent.pkl", "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"RL agent not found, using direct lookup: {e}")
            return None
    
    def load_hash_lookup(self):
        """Load hash to servo lookup table"""
        try:
            with open("hash_to_servo_lookup.json", "r") as f:
                data = json.load(f)
                return data.get('hash_to_servo_lookup', {})
        except Exception as e:
            logging.error(f"Failed to load hash lookup: {e}")
            raise
    
    def plan_optimal_path(self, target_hash):
        """Plan optimal path from current to target position"""
        target_angles = self.hash_lookup.get(target_hash)
        
        if not target_angles:
            logging.warning(f"Target hash {target_hash} not found in lookup")
            return None
        
        if self.rl_agent:
            path = self.rl_agent.plan_path(self.current_position, target_angles)
            return path
        else:
            return [target_angles]
    
    def handle_prediction_message(self, message):
        """Process ML prediction and plan servo movement"""
        try:
            data = json.loads(message['data'])
            predicted_hash = data['predicted_hash']
            
            servo_path = self.plan_optimal_path(predicted_hash)
            
            if servo_path:
                command_message = {
                    "timestamp": datetime.now().isoformat(),
                    "servo_path": servo_path,
                    "target_hash": predicted_hash,
                    "source": "rl-planner"
                }
                
                self.redis_client.publish(
                    self.output_channel,
                    json.dumps(command_message)
                )
                
                logging.info(f"Servo path planned for hash: {predicted_hash}")
            
        except Exception as e:
            logging.error(f"Error processing prediction message: {e}")
    
    def run(self):
        """Main service loop"""
        logging.info("Starting RL planning service...")
        
        self.pubsub.subscribe(self.input_channel)
        
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                self.handle_prediction_message(message)

if __name__ == "__main__":
    service = RLPlanningService()
    service.run()