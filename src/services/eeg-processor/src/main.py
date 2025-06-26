# EEG Processing Service Entry Point

import os
import json
import numpy as np
from models.inference_model import InferenceModel
from utils.preprocessing import preprocess_data

def load_configuration():
    config_path = os.path.join(os.path.dirname(__file__), '../../shared/config/mechanicus_run_configuration.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_inference_model():
    model_path = os.path.join(os.path.dirname(__file__), '../../shared/models/inference_model.pkl')
    return InferenceModel.load(model_path)

def main():
    config = load_configuration()
    model = load_inference_model()

    # Initialize EEG data processing
    while True:
        # Simulate reading EEG data
        raw_eeg_data = np.random.normal(size=(config['dataset']['n_eeg_channels'],))
        
        # Preprocess the data
        processed_data = preprocess_data(raw_eeg_data)

        # Perform inference
        position_hash = model.predict(processed_data)

        # Output the position hash
        print(f'Predicted Position Hash: {position_hash}')

if __name__ == "__main__":
    main()