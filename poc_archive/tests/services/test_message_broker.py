import unittest
import redis
import json
import time
from eeg_data_publisher_service import EEGDataPublisherService

class TestMessageBroker(unittest.TestCase):
    
    def setUp(self):
        self.redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
        self.test_channel = "test_eeg_channel"
    
    def test_eeg_publisher_basic(self):
        """Test basic EEG data publishing"""
        publisher = EEGDataPublisherService(channel=self.test_channel)
        
        sample = publisher.generate_eeg_sample()
        success = publisher.publish_sample(sample)
        
        self.assertTrue(success)
        self.assertIn('eeg_data', sample)
        self.assertEqual(len(sample['eeg_data']), 5)  # Default 5 channels
    
    def test_eeg_publisher_integration(self):
        """Test integration with Redis subscriber"""
        publisher = EEGDataPublisherService(channel=self.test_channel, sampling_rate=10)
        
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.test_channel)
        
        for _ in range(3):
            sample = publisher.generate_eeg_sample()
            publisher.publish_sample(sample)
            time.sleep(0.1)
        
        messages = []
        for _ in range(3):
            message = pubsub.get_message(timeout=1.0)
            if message and message['type'] == 'message':
                data = json.loads(message['data'])
                messages.append(data)
        
        self.assertEqual(len(messages), 2)
        for msg in messages:
            self.assertIn('eeg_data', msg)
            self.assertIn('timestamp', msg)