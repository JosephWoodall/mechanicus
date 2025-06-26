import redis 
import json 
import numpy 
import time 
import logging
import argparse 
import random 
from datetime import datetime 
from typing import Dict, Any, Optional 

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eeg_data_publisher_service.py")

class EEGDataPublisherService:
    """Simulated EEG data publisher for testing Mechanicus services.
    
    Generates realistic EEG data with configurable anomaly spikes and publishes it to Redis channels for downstream service testing.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", channel: str = "eeg_anomalies", n_channels: int = 5, sampling_rate: float = 100.0, anomaly_rate: float = 0.05):
        """Initialize simulated EEG data publisher service

        Args:
            redis_url (str, optional): Redis connection URL. Defaults to "redis://localhost:6379".
            channel (str, optional): Redis channel which to publish. Defaults to "eeg_anomalies".
            n_channels (int, optional): number of eeg channels to simulate. Defaults to 5.
            sampling_rate (float, optional): sampling rate in Hz. Defaults to 100.0.
            anomaly_rate (float, optional): probability of generating anomaly spike (0-1). Defaults to 0.05.
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.channel = channel 
        self.n_channels = n_channels 
        self.sampling_rate = sampling_rate
        self.anomaly_rate = anomaly_rate
        self.sample_interval = 1.0 / self.sampling_rate
        
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        self.anomaly_multiplier = 3.0
        
        self.total_samples = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        
        logger.info(f"EEGDataPublisherService Initialized:")
        logger.info(f"  Redis URL: {self.redis_client}")
        logger.info(f"  Channel: {self.channel}")
        logger.info(f"  EEG Channels: {self.n_channels}")
        logger.info(f"  Sampling Rate: {self.sampling_rate} Hz")
        logger.info(f"  Anomaly Rate: {self.anomaly_rate * 100:.2f}%")
        
    def generate_baseline_eeg(self) -> numpy.ndarray:
        """Generate baseline EEG data (normal brain activity).

        Returns:
            numpy.ndarray: numpy array of shape (n_channels,) with baseline EEG values.
        """
        eeg_data = numpy.random.normal(
            self.baseline_mean,
            self.baseline_std,
            self.n_channels
        )
        time_factor = time.time() * 2 * numpy.pi 
        for i in range(self.n_channels):
            frequency = 8 + (i * 3)
            amplitude = 0.3 
            eeg_data[i] += amplitude * numpy.sin(frequency * time_factor)
        
        return eeg_data
    
    def generate_anomaly_eeg(self) -> numpy.ndarray:
        """
        Generate EEG data with anomaly spike.
        
        Returns:
            numpy array of shape (n_channels,) with anomalous EEG values
        """
        baseline = self.generate_baseline_eeg()
        
        # Create spike in random subset of channels
        spike_channels = random.sample(
            range(self.n_channels), 
            random.randint(1, max(1, self.n_channels // 2))
        )
        
        for channel in spike_channels:
            # Generate anomaly spike
            spike_magnitude = random.uniform(2.5, 5.0)  # 2.5-5x std deviation
            spike_direction = random.choice([-1, 1])    # Positive or negative spike
            baseline[channel] += spike_direction * spike_magnitude * self.baseline_std
        
        return baseline
    
    def generate_eeg_sample(self) -> Dict:
        """Generate a single EEG sample with or without anomaly spike.

        Returns:
            Dict: dictionary containing EEG data and metadata.
        """
        is_anomaly = random.random() < self.anomaly_rate 
        
        if is_anomaly:
            eeg_data = self.generate_anomaly_eeg()
            self.anomaly_count += 1
        else:
            eeg_data = self.generate_baseline_eeg()
        
        self.total_samples += 1
        
        sample = {
            'timestamp': datetime.now().isoformat(),
            'sample_id': f"eeg_sample_{self.total_samples:06d}",
            'eeg_data': eeg_data.tolist(),
            'n_channels': self.n_channels,
            'is_anomaly': is_anomaly,
            'source' : 'test_EEGDataPublisherService',
            'metadata': {
                "baseline_mean": self.baseline_mean,
                "baseline_std": self.baseline_std,
                "total_samples": self.total_samples,
                "anomaly_count": self.anomaly_count,
                "anomaly_rate": self.anomaly_count / self.total_samples if self.total_samples > 0 else 0
            }
        }
        
        return sample
    
    def publish_sample(self, sample: Dict) -> bool:
        """Publish EEG sample to Redis channel.

        Args:
            sample (Dict): EEG sample dictionary.

        Returns:
            bool: True if published successfully, False otherwise.
        """
        try:
            message = json.dumps(sample)
            self.redis_client.publish(self.channel, message)
            return True 
        except Exception as e:
            logger.error(f"Failed to publish sample: {e}")
            return False
        
    def print_statistics(self):
        """Print current streaming statistics.
        """
        runtime = time.time() - self.start_time 
        actual_rate = self.total_samples / runtime if runtime > 0 else 0 
        actual_anomaly_rate = self.anomaly_count / self.total_samples if self.total_samples > 0 else 0
        
        logger.info(f"Streaming Statistics:")
        logger.info(f"  - Runtime: {runtime:.1f}s")
        logger.info(f"  - Total Samples: {self.total_samples}")
        logger.info(f"  - Anomalies: {self.anomaly_count}")
        logger.info(f"  - Target Rate: {self.sampling_rate:.1f} Hz")
        logger.info(f"  - Actual Rate: {actual_rate:.1f} Hz")
        logger.info(f"  - Target Anomaly Rate: {self.anomaly_rate * 100:.1f}%")
        logger.info(f"  - Actual Anomaly Rate: {actual_anomaly_rate * 100:.1f}%")
        
    def run_continuously(self, duration: Optional[float] = None, verbose: bool = True):
        """Run the EEG data publisher continuously, generating and publishing samples at the configured sampling rate.

        Args:
            duration (Optional[float], optional): duration to run in seconds (none = indefinite). Defaults to None.
            verbose (bool, optional): whether to print periodic statistics. Defaults to True.
        """
        logger.info(f"Starting EEG data streaming to channel '{self.channel}.")
        logger.info(f"Streaming at {self.sampling_rate} Hz with {self.anomaly_rate * 100:.1f}% anomaly rate.")
        
        if duration:
            logger.info(f"Running for {duration:.1f} seconds.")
        else:
            logger.info("Running indefinitely until interrupted.")
        
        start_time = time.time()
        last_stats_time = start_time 
        
        try:
            while True:
                loop_start = time.time()
                
                sample = self.generate_eeg_sample()
                success = self.publish_sample(sample)
                
                if not success:
                    logger.warning("Failed to publish sample, continuing...")
                    
                if verbose and (time.time() - last_stats_time) >= 10.0:
                    self.print_statistics()
                    last_stats_time = time.time()
                
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"Completed {duration:.1f} seconds of streaming.")
                    break 
                
                elapsed = time.time() - loop_start 
                sleep_time = max(0, self.sample_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user.")
        
        self.print_statistics()
        logger.info("EEG data streaming completed.")
        
    def run_batch(self, n_samples: int, batch_interval: float = 1.0):
        """Run batch mode - send N samples then wait.

        Args:
            n_samples (int): number of samples per batch.
            batch_interval (float, optional): interval between batches in seconds. Defaults to 1.0.
        """
        
        logger.info(f"Starting batch mode: {n_samples} samples every {batch_interval:.1f} seconds.")
        
        try:
            while True:
                batch_start = time.time()
                
                logger.info(f"Sending batch of {n_samples} samples...")
                
                for i in range(n_samples):
                    sample = self.generate_eeg_sample()
                    self.publish_sample(sample)
                
                batch_time = time.time() - batch_start 
                logger.info(f"batch sent in {batch_time:.2f} seconds.")
                
                time.sleep(batch_interval)
        except KeyboardInterrupt:
            logger.info("Batch mode interrupted by user.")
            
        self.print_statistics()
        
def main():
    """Main function with CLI interface.
    """
    parser = argparse.ArgumentParser(description = "EEG Data Publisher Service for Mechanicus Testing", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
    "--redis-url", 
    default="redis://localhost:6379",
    help="Redis connection URL"
)

    parser.add_argument(
        "--channel", 
        default="eeg_anomalies",
        help="Redis channel to publish to"
    )
    
    parser.add_argument(
        "--channels", 
        type=int, 
        default=5,
        help="Number of EEG channels to simulate"
    )
    
    parser.add_argument(
        "--rate", 
        type=float, 
        default=100.0,
        help="Sampling rate in Hz"
    )
    
    parser.add_argument(
        "--anomaly-rate", 
        type=float, 
        default=0.05,
        help="Anomaly rate (0.0-1.0)"
    )
    
    parser.add_argument(
        "--duration", 
        type=float,
        help="Duration to run in seconds (default: indefinite)"
    )
    
    parser.add_argument(
        "--batch-mode", 
        action="store_true",
        help="Run in batch mode instead of continuous"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=10,
        help="Number of samples per batch (batch mode only)"
    )
    
    parser.add_argument(
        "--batch-interval", 
        type=float, 
        default=1.0,
        help="Interval between batches in seconds (batch mode only)"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress periodic statistics output"
    )
    
    args = parser.parse_args()
    
    if not (0.0 <= args.anomaly_rate <= 1.0):
        parser.error("Anomaly rate must be betwen 0.0 and 1.0")
        
    if args.rate <= 0.0:
        parser.error("Sampling rate must be positive")
    
    publisher = EEGDataPublisherService(
        redis_url=args.redis_url,
        channel=args.channel,
        n_channels=args.channels,
        sampling_rate=args.rate,
        anomaly_rate=args.anomaly_rate
    )
    
    try:
        publisher.redis_client.ping()
        logger.info("Connected to Redis server successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis server: {e}")
        return 1

    if args.batch_mode:
        publisher.run_batch(args.batch_size, args.batch_interval)
    else:
        publisher.run_continuously(args.duration, verbose = not args.quiet)
        
    return 0

if __name__ == "__main__":
    exit(main())