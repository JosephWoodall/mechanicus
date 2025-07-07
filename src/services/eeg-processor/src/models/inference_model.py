import joblib
import redis
import json
import logging
import warnings
import os
import pandas
import time
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


class PreprocessData:

    @staticmethod
    def preprocess_data(
        data: dict
    ) -> pandas.DataFrame:
        # logging.info("Preprocessing of the EEG data for model_input")
        try:
            eeg_values = None

            if isinstance(data, dict):
                if 'eeg_data' in data:
                    eeg_values = data['eeg_data']
                    # logging.info("Extracted eeg_data from top level.")
                elif 'original_sample' in data and isinstance(data['original_sample'], dict):
                    if 'eeg_data' in data['original_sample']:
                        eeg_values = data['original_sample']['eeg_data']
                        # logging.info(
                        #    f"Extracted eeg_data from original_sample: {eeg_values}")
                else:
                    numeric_values = [
                        v for v in data.values() if isinstance(v, (int, float))]
                    if numeric_values:
                        eeg_values = numeric_values
                        logging.info(
                            f"Using numeric values as eeg_data: {eeg_values}")
                    else:
                        raise ValueError("No eeg_data found in input data")
            elif isinstance(data, list):
                eeg_values = data
                logging.info("Input data is a list; using as eeg_data.")
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            if eeg_values is None:
                raise ValueError("No eeg_data found in input data.")

            column_names = [f'eeg_channel_{i}' for i in range(len(eeg_values))]
            df = pandas.DataFrame([eeg_values], columns=column_names)

            # logging.info(f"Created DataFrame with shape: {df.shape}")
            # logging.info(f"DataFrame columns: {df.columns.tolist()}")
            # logging.info(f"DataFrame values: {df.iloc[0].tolist()}")

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            x = pandas.DataFrame(scaled_data, columns=df.columns)

            # logging.info(f"Scaled EEG data: {x.iloc[0].tolist()}")
            # logging.info(
            #    "...Preprocessing of the EEG data for model input complete.")

            return x

        except Exception as e:
            logging.error(
                f"An error has occurred when preprocessing the EEG data for model input: {e}")
            return pandas.DataFrame()


class EEGInferenceModel:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.model_loaded = False
        self.last_model_check = 0
        self.model_check_interval = 30

        redis_host = 'redis' if os.getenv('REDIS_URL') else 'localhost'
        self.redis_client = redis.Redis(
            host=redis_host, port=6379, decode_responses=True)

        logger.info("=" * 50)
        logger.info(f"EEGInferenceModel Initialized:")
        logger.info(f"  Model Path: {self.model_path}")
        logger.info(f"  Input Channel: eeg_data_processed")
        logger.info(f"  Output Channel: predicted_servo_angles")
        logger.info("=" * 50)

        self.try_load_model()

    def try_load_model(self):
        """Try to load the model, return True if successful"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)['model']
                logger.info(
                    f"Successfully loaded model from {self.model_path}")
                logger.info(f"Model type: {type(self.model)}")
                self.model_loaded = True
                return True
            except Exception as e:
                logger.error(
                    f"Failed to load model from {self.model_path}: {e}")
                self.model = None
                self.model_loaded = False
                return False
        else:
            if not self.model_loaded:
                logging.warning("=" * 50)
                logger.warning(f"Model file not found at {self.model_path}")
                logger.warning(f"Current working directory: {os.getcwd()}")
                logger.warning(
                    f"Files in /app/shared/models/: {os.listdir('/app/shared/models/') if os.path.exists('/app/shared/models/') else 'Directory does not exist'}")
                logger.warning("Waiting for model to become available...")
                logger.warning(
                    "---------> To create a model, run: docker-compose -f docker-compose.offline_training.yml up --build in a separate terminal.\nThis will train a model and save it to /app/shared/models/inference_model.pkl\nThis inference script will automatically load the model when it appears, and simply pass until you do so.\nNo servo executions will be taken until inference model is found!")
                logging.warning("=" * 50)
            self.model = None
            self.model_loaded = False
            return False

    def check_for_model_periodically(self):
        """Periodically check if model has become available"""
        current_time = time.time()
        if current_time - self.last_model_check > self.model_check_interval:
            self.last_model_check = current_time
            if not self.model_loaded:
                if self.try_load_model():
                    logger.info(
                        "Model is now available! Starting predictions...")
                else:
                    logger.info("Still waiting for model file to appear...")

    def preprocess_data(self, eeg_data):
        """Preprocess EEG data for model prediction"""
        preprocessed_dataframe = PreprocessData.preprocess_data(eeg_data)

        if preprocessed_dataframe.empty:
            logger.error("Preprocessing returned empty DataFrame")
            return None

        preprocessed_array = preprocessed_dataframe.values

        # logger.info(
        #    f"Final preprocessed data shape: {preprocessed_array.shape}")
        # logger.info(f"Final preprocessed data: {preprocessed_array[0]}")

        return preprocessed_array

    def predict_servo_angles(self, processed_data):
        """Make prediction on preprocessed EEG data and return servo angles"""
        if self.model is None:
            return None

        try:
            # logger.info(
            #   f"Making prediction with processed data shape: {processed_data.shape}")

            prediction = self.model.predict(processed_data)

            if prediction.ndim > 1:
                servo_angles = prediction[0]
            else:
                servo_angles = prediction

            # logger.info(f"Predicted servo angles: {servo_angles}")
            return servo_angles
        except Exception as e:
            # logger.error(f"Error making servo angle prediction: {e}")
            return None

    def publish_servo_angles(self, servo_angles, original_data):
        """Publish predicted servo angles to Redis channel"""
        if servo_angles is not None:
            result = {
                'servo_angles': servo_angles.tolist() if hasattr(servo_angles, 'tolist') else servo_angles,
                'timestamp': original_data.get('timestamp') if isinstance(original_data, dict) else None,
                'original_data_id': original_data.get('sample_id') if isinstance(original_data, dict) and 'original_sample' in original_data else None
            }

            try:
                self.redis_client.publish(
                    'predicted_servo_angles', json.dumps(result))
                # logger.info(
                #     f"Published servo angles to predicted_servo_angles channel: {result['servo_angles']}")
            except Exception as e:
                logger.error(f"Failed to publish servo angles: {e}")

    def monitor_redis_channel(self, channel_name='eeg_data_processed'):
        """Monitor Redis channel, log data, and make servo angle predictions"""
        logger.info(f"Starting to monitor Redis channel: {channel_name}")

        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(channel_name)

        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    self.check_for_model_periodically()

                    try:
                        data = json.loads(message['data'])
                        # logger.info(f"Received EEG data from Redis")

                        if self.model is not None:
                            processed_data = self.preprocess_data(data)

                            if processed_data is not None:
                                # logger.info(
                                #    f"Preprocessed data shape: {processed_data.shape}")

                                servo_angles = self.predict_servo_angles(
                                    processed_data)

                                if servo_angles is not None:
                                    self.publish_servo_angles(
                                        servo_angles, data)
                            else:
                                logger.error("Failed to preprocess data")
                        else:
                            pass

                    except json.JSONDecodeError:
                        # logger.info(f"Raw data (non-JSON): {message['data']}")
                        if self.model is not None:
                            try:
                                raw_data = message['data'].strip()
                                if raw_data:
                                    numeric_data = [
                                        float(x) for x in raw_data.split(',')]
                                    processed_data = self.preprocess_data(
                                        {'eeg_data': numeric_data})

                                    if processed_data is not None:
                                        servo_angles = self.predict_servo_angles(
                                            processed_data)
                                        if servo_angles is not None:
                                            self.publish_servo_angles(
                                                servo_angles, {'raw_data': raw_data})
                            except Exception as e:
                                logger.error(f"Error processing raw data: {e}")

                elif message['type'] == 'subscribe':
                    logger.info(f"Subscribed to channel: {channel_name}")

        except KeyboardInterrupt:
            logger.info("Stopping monitoring...")
        finally:
            pubsub.unsubscribe(channel_name)
            pubsub.close()


if __name__ == "__main__":
    logger.info("=== DEBUGGING FILE SYSTEM ===")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(
        f"Contents of /app: {os.listdir('/app') if os.path.exists('/app') else 'Does not exist'}")
    logger.info(
        f"Contents of /app/shared: {os.listdir('/app/shared') if os.path.exists('/app/shared') else 'Does not exist'}")
    logger.info(
        f"Contents of /app/shared/models: {os.listdir('/app/shared/models') if os.path.exists('/app/shared/models') else 'Directory does not exist'}")

    model_path = '/app/shared/models/inference_model.pkl'
    logger.info(f"Trying to load model from: {model_path}")
    logger.info(f"Model file exists: {os.path.exists(model_path)}")

    model = EEGInferenceModel(model_path)
    model.monitor_redis_channel('eeg_data_processed')
