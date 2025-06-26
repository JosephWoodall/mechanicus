import redis
import json
import logging
import numpy as np
from datetime import datetime
import time

class EEGMonitorService:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.output_channel = "eeg_anomalies"
        
    def detect_anomaly(self, eeg_data):
        """Your anomaly detection logic"""
        threshold = 2.5  
        return np.any(np.abs(eeg_data) > threshold)
    
    def publish_anomaly(self, eeg_data):
        """Publish EEG anomaly to message broker"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "eeg_data": eeg_data.tolist(),
            "source": "eeg-monitor",
            "anomaly_detected": True
        }
        
        self.redis_client.publish(
            self.output_channel, 
            json.dumps(message)
        )
        logging.info(f"Published EEG anomaly to channel: {self.output_channel}")
    
    def run(self):
        """Main monitoring loop"""
        logging.info("Starting EEG monitoring service...")
        
        while True:
            eeg_data = ""  # Replace with actual EEG data collection logic
            
            if self.detect_anomaly(eeg_data):
                self.publish_anomaly(eeg_data)
            
            time.sleep(0.01)  # 100Hz sampling

if __name__ == "__main__":
    service = EEGMonitorService()
    service.run()