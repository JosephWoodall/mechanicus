import joblib
import numpy

class EEGInferenceModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, eeg_data):
        processed_data = self.preprocess_data(eeg_data)
        prediction = self.model.predict(processed_data)
        return prediction

    def preprocess_data(self, eeg_data):
        # Assuming eeg_data is a 2D array where rows are samples and columns are features
        # Normalize or scale the data as needed
        return numpy.array(eeg_data)  # Placeholder for actual preprocessing logic

# Example usage
if __name__ == "__main__":
    model = EEGInferenceModel('../shared/models/inference_model.pkl')
    sample_eeg_data = numpy.random.rand(1, 5)  # Replace with actual EEG data
    prediction = model.predict(sample_eeg_data)
    print("Predicted Position Hash:", prediction)
    
    