import redis
import json
import logging
import pickle
import pandas as pd
from datetime import datetime

class MLInferenceService:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.input_channel = "eeg_anomalies"
        self.output_channel = "ml_predictions"
        
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained ML model"""
        try:
            with open("inference_model.pkl", "rb") as f:
                data = pickle.load(f)
                return data["model"]
        except Exception as e:
            logging.error(f"Failed to load ML model: {e}")
            raise
    
    def preprocess_eeg_data(self, eeg_data):
        """Preprocess EEG data for ML inference"""
        eeg_features = {f'eeg_{i}': val for i, val in enumerate(eeg_data)}
        df = pd.DataFrame([eeg_features])
        return df
    
    def predict_position_hash(self, eeg_data):
        """Predict position hash from EEG data"""
        preprocessed_data = self.preprocess_eeg_data(eeg_data)
        prediction = self.model.predict(preprocessed_data)[0]
        return prediction
    
    def handle_eeg_message(self, message):
        """Process incoming EEG anomaly message"""
        try:
            data = json.loads(message['data'])
            eeg_data = data['eeg_data']
            
            # ML Inference
            predicted_hash = self.predict_position_hash(eeg_data)
            
            # Publish prediction
            prediction_message = {
                "timestamp": datetime.now().isoformat(),
                "predicted_hash": predicted_hash,
                "source_timestamp": data['timestamp'],
                "source": "ml-inference"
            }
            
            self.redis_client.publish(
                self.output_channel,
                json.dumps(prediction_message)
            )
            
            logging.info(f"ML prediction published: {predicted_hash}")
            
        except Exception as e:
            logging.error(f"Error processing EEG message: {e}")
    
    def run(self):
        """Main service loop"""
        logging.info("Starting ML inference service...")
        
        self.pubsub.subscribe(self.input_channel)
        
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                self.handle_eeg_message(message)

if __name__ == "__main__":
    service = MLInferenceService()
    service.run()