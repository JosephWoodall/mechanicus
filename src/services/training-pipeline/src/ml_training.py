from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

class MLTraining:
    def __init__(self, training_data_path, model_output_path):
        self.training_data_path = training_data_path
        self.model_output_path = model_output_path
        self.model = RandomForestClassifier()

    def load_data(self):
        if os.path.exists(self.training_data_path):
            data = pd.read_json(self.training_data_path)
            X = data.drop('target', axis=1)
            y = data['target']
            return X, y
        else:
            raise FileNotFoundError(f"Training data not found at {self.training_data_path}")

    def train_model(self, X, y):
        self.model.fit(X, y)

    def save_model(self):
        joblib.dump(self.model, self.model_output_path)

    def run(self):
        X, y = self.load_data()
        self.train_model(X, y)
        self.save_model()

if __name__ == "__main__":
    training_data_path = os.path.join(os.path.dirname(__file__), 'training_data.json')
    model_output_path = os.path.join(os.path.dirname(__file__), 'inference_model.pkl')
    
    ml_trainer = MLTraining(training_data_path, model_output_path)
    ml_trainer.run()