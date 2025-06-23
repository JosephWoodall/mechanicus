import pandas
import numpy
import os
import json

from contextlib import redirect_stdout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

import time
import datetime
import pickle
from enum import Enum

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="mechanicus_training.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

logger.info("Importing necessary libraries...")


import warnings

warnings.filterwarnings("ignore")

logging.info("...Library import complete.")

class DataCollector:
    
    @staticmethod
    def load_servo_eeg_data(json_filename: str = None) -> pandas.DataFrame:
        """Load servo EEG data from JSON file and convert to DataFrame.
        
        Args:
            json_filename (str): Path to the JSON file containing servo EEG data
            
        Returns:
            pandas.DataFrame: DataFrame with servo angles, positions, and EEG data
        """
        logging.info(f"Loading servo EEG data from {json_filename}...")
        
        try:
            with open(json_filename, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            metadata = data.get('metadata', {})
            logging.info(f"Dataset metadata: {metadata}")
            
            # Extract sample data (exclude metadata)
            samples = {k: v for k, v in data.items() if k != 'metadata'}
            
            # Convert to DataFrame
            rows = []
            for sample_id, sample_data in samples.items():
                row = {
                    **{f'servo_angles_{i}': angle for i, angle in enumerate(sample_data['servo_angles'])},
                    **{f'position_{coord}': pos for coord, pos in zip(['x', 'y', 'z'], sample_data['position'])},
                    'position_hash': sample_data['position_hash'],
                    **{f'eeg_{i}': eeg_val for i, eeg_val in enumerate(sample_data['eeg_data'])}
                }
                rows.append(row)
            
            df = pandas.DataFrame(rows)
            
            #df['servo_angle_combination'] = df.apply(
            #    lambda row: f"{row['servo_angles_0']:.1f}_{row['servo_angles_1']:.1f}_{row['servo_angles_2']:.1f}", 
            #    axis=1
            #)
            
            logging.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
            #logging.info(f"Unique servo angle combinations: {df['servo_angle_combination'].nunique()}")
            
            logging.info("...Servo EEG data loaded successfully.")
            logging.info(f"Dataframe: {df.head()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading servo EEG data from {json_filename}: {e}")
            raise

class ExploratoryDataAnalysis:

    @staticmethod
    def get_summary_statistics(
        focus_data: pandas.DataFrame, filename: str = "eda.txt"
    ) -> None:
        logger.info("Performing exploratory data analysis on the focused data...")

        try:
            rel_path_filename = rf"{filename}"
            with open(rel_path_filename, "w") as f:
                with redirect_stdout(f):
                    print("-" * 100)
                    print(f"{filename}")
                    print("-" * 100)
                    print("-" * 25)
                    print("Info")
                    print("-" * 25)
                    print(focus_data.info())
                    print("\n")
                    print("-" * 25)
                    print("Describe")
                    print("-" * 25)
                    print(focus_data.describe())
                    print("-" * 25)
                    print("\n")
                    print("-" * 25)
                    print("Nulls per Feature")
                    print("-" * 25)
                    total_nulls = focus_data.isnull().sum(axis=0)
                    percent_nulls = (focus_data.isnull().sum() / len(focus_data)) * 100
                    null_summary = pandas.DataFrame(
                        {
                            "total_null_values": total_nulls,
                            "percent_of_null_values": percent_nulls,
                        }
                    )
                    print(null_summary)
                    print("Features with more than 10% null values:")
                    high_null_features = null_summary[
                        null_summary["percent_of_null_values"] > 10
                    ]
                    print(high_null_features)
                    print("-" * 100)
            logging.info(
                rf"...Exploratory data analysis completed, results saved to: {rel_path_filename}"
            )
        except Exception as e:
            logging.error(
                f"An error has occured when performing exploratory data analysis: {e}"
            )


class Phase(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class PreprocessData:
    
    @staticmethod
    def preprocess_data(
        focus_data: pandas.DataFrame, response_variable: str = None, phase=Phase
    ) -> tuple:
        logging.info("Preprocessing of the focus data for model_input")
        try:
            categorical_features = focus_data.select_dtypes(
                include=["category"]
            ).columns
            for col in categorical_features:
                le = LabelEncoder()
                focus_data[col] = le.fit_transform(focus_data[col])

            scaler = StandardScaler()

            focus_data.columns = focus_data.columns.astype(str)

            if phase == Phase.TRAINING:

                x = focus_data.drop(response_variable, axis=1)
                # x = focus_data.drop("prosthetic_cartesian_3d_position", axis=1)
                x_feature_names = x.columns
                x = scaler.fit_transform(x)
                x = pandas.DataFrame(x, columns=x_feature_names)
                y = focus_data[response_variable]
            elif phase == Phase.INFERENCE:
                x = focus_data.drop(response_variable, axis=1)
                # x = focus_data.drop("prosthetic_cartesian_3d_position", axis=1)
                x_feature_names = x.columns
                x = scaler.fit_transform(x)
                x = pandas.DataFrame(x, columns=x_feature_names)
                y = None
            else:
                raise ValueError(
                    f"Invalid phase: {phase}. Expected Phase('training') or Phase('inference')"
                )
            logging.info("...Preprocessing of the focus data for model input complete.")
        except Exception as e:
            logging.error(
                f"An error has occured when preprocessing the focus data for model input: {e}"
            )

        return x, y


class TrainModel:
    @staticmethod
    def evaluateModel(
        x: tuple = None,
        y: tuple = None,
        cv: int = 10,
        scoring_metric: str = "accuracy",
        filename: str = "mechanicusModelOutput.txt",
    ):
        logging.info(
            "Evaluating the model population and returning the best performing model..."
        )
        try:
            rel_path_filename = rf"{filename}"
            with open(rel_path_filename, "w") as f:
                with redirect_stdout(f):
                    try:
                        start_time = time.time()
                        print("-" * 100)
                        print("Models to be tested")
                        print("-" * 100)

                        models = [
                            ("Random Forest Classifier", RandomForestClassifier()),
                        ]

                        def generateRandomIntList(length_of_list: int = 3):
                            random_numbers = []
                            for _ in range(length_of_list):
                                random_number = numpy.random.randint(0, 100)
                                random_numbers.append(random_number)
                            return random_numbers

                        def generateRandomFloatList(length_of_list: int = 3):
                            return numpy.random.uniform(0, 100, size=length_of_list)

                        param_grid = {
                            
                            "Random Forest Classifier": {
                                #'n_estimators':generateRandomIntList()
                                # , 'max_depth':generateRandomIntList()
                                # , 'min_samples_split':generateRandomIntList()
                                "max_features": ["sqrt", "log2"]
                                # , 'max_leaf_nodes':generateRandomIntList()
                                # , 'ccp_alpha':generateRandomFloatList()
                            },
                        }

                        best_models = {}

                        for name, model in models:
                            grid_search = RandomizedSearchCV(
                                estimator=model,
                                param_distributions=param_grid[name],
                                cv=cv,
                                scoring=scoring_metric,
                            )
                            grid_search.fit(x, y)

                            best_model = grid_search.best_estimator_
                            best_models[name] = best_model

                        results = {}

                        for name, model in best_models.items():
                            print("\n")
                            print("-" * 25)
                            print(f"{name}")
                            print("-" * 25)
                            scores = cross_val_score(
                                model, x, y, cv=cv, scoring=scoring_metric
                            )
                            results[name] = {
                                "model": model,
                                "mean_score": scores.mean(),
                                "std_score": scores.std(),
                                "hyperparameters": model.get_params(),
                                "scores": scores.tolist(),
                            }
                            print(
                                f"{name}: Mean Score = {results[name]['mean_score']:.4f} (+/- {results[name]['std_score']:.4f})"
                            )

                            selector = SelectFromModel(model).fit(x, y)
                            feature_scores = []
                            if name in [
                                "Random Forest Classifier",
                                "Ada Boost Classifier",
                                "Extra Trees Classifier",
                                "Gradient Boosting Classifier",
                            ]:
                                feature_scores = pandas.DataFrame(
                                    {
                                        "features": x.columns,
                                        "scores": selector.estimator_.feature_importances_,
                                    }
                                )
                            else:
                                feature_scores = pandas.DataFrame(
                                    {
                                        "features": x.columns,
                                        "scores": selector.estimator_.coef_,
                                    }
                                )
                            feature_scores = feature_scores.sort_values(
                                "scores", ascending=False
                            )

                            print("Important Features")
                            print(feature_scores)
                            print("-" * 100)

                        best_score = None
                        best_hyperparameters = None
                        best_score_key = None
                        best_model_for_inference = None

                        for key, value in results.items():
                            model = value.get("model")
                            mean_score = value.get("mean_score")
                            hyperparameters = value.get("hyperparameters")
                            if mean_score is not None and (
                                best_score is None or mean_score > best_score
                            ):
                                best_model_for_inference = model
                                best_score = mean_score
                                best_score_key = key
                                best_hyperparameters = hyperparameters
                        print("\n")
                        print("-" * 25)
                        print(f"Best Model Proposed to be Used for Inference:")
                        print("-" * 25)
                        print(f"Model Class: {best_model_for_inference}")
                        print(f"Largest Mean Score: {best_score}")
                        print(f"Best Score Key: {best_score_key}")
                        print(f"Best Score Hyperparameters: {best_hyperparameters}")
                        print("-" * 25)

                        logging.info(
                            rf"...Model population has been evaluated, results saved to: {rel_path_filename}."
                        )
                        relative_path = "src"
                        model_filename = r"inference_model.pkl"
                        data_to_pickle = {
                            "model": best_model_for_inference,
                            "largest_mean_score": best_score,
                            "train_data_size": len(x),
                            "date_created": datetime.datetime.now().isoformat(),
                        }
                        logging.info("Executing Model Comparison...")

                        print("\n")
                        print("-" * 100)
                        print("Executing Model Comparison...")
                        print(
                            rf"If no model is saved via Pickle to {relative_path}/{model_filename}, then Best Model for Inference is saved to Pickle and used for inference."
                        )
                        print(
                            rf"If existing model is saved via Pickle to {relative_path}/{model_filename}, then will compare Largest Mean Score of existing model to largest mean score of Best Model for Inference."
                        )
                        print("-" * 100)

                        if not os.path.isfile(rf"{relative_path}/{model_filename}"):
                            print("No saved model found...")
                            print("Saving Best Model for Inference via Pickle...")
                            try:
                                logging.info(
                                    f"Saving best performing model to {model_filename}..."
                                )
                                with open(rf"{model_filename}", "wb") as f:
                                    pickle.dump(data_to_pickle, f)
                                logging.info(
                                    rf"...Best performing model saved for inference to {model_filename}."
                                )
                            except Exception as e:
                                logging.error(
                                    f"Inference model has not been saved because of the following error: {e}."
                                )
                            print("...Saved Best Model for Inference to Pickle.")

                        elif os.path.isfile(rf"{filename}"):
                            print("Existing saved model found...")
                            with open(rf"{model_filename}", "rb") as f:
                                data_from_pickle = pickle.load(f)
                            data_from_pickle_model = data_from_pickle["model"]
                            data_from_pickle_largest_mean_score = data_from_pickle[
                                "largest_mean_score"
                            ]
                            data_from_pickle_train_data_size = data_from_pickle[
                                "train_data_size"
                            ]
                            data_from_pickle_date_created = data_from_pickle[
                                "date_created"
                            ]
                            print("Comparing Saved Model and Newly Trained Model...")
                            print("-" * 25)
                            print("Existing Inference Model Information")
                            print("-" * 25)
                            print(
                                f"Existing Inference Model Type: {data_from_pickle_model}"
                            )
                            print(
                                f"Existing Inference Model Largest Mean Score: {data_from_pickle_largest_mean_score}"
                            )
                            print(
                                f"Existing Inference Model Train Data Size Used: {data_from_pickle_train_data_size}"
                            )
                            print(
                                f"Existing Inference Model Date Created: {data_from_pickle_date_created}"
                            )
                            print("-" * 25)
                            print("Newly Trained Model Information")
                            print("-" * 25)
                            print(
                                f"Newly Trained Model Type: {best_model_for_inference}"
                            )
                            print(
                                f"Newly Trained Model Largest Mean Score: {best_score}"
                            )
                            print(f"Newly Trained model Train Data Size Used: {len(x)}")
                            print(
                                f"Newly Trained Model Date Created: {datetime.datetime.now().isoformat()}"
                            )
                            if best_score >= data_from_pickle_largest_mean_score:
                                print(
                                    "Newly Trained model better than Existing Pickled Model..."
                                )
                                print("Saving BEst Model for Inference to PIckle...")
                                with open(rf"{model_filename}", "wb") as f:
                                    pickle.dump(data_from_pickle, f)
                                print("...Saved Best Model for Inference to Pickle")
                            elif best_score < data_from_pickle_largest_mean_score:
                                print(
                                    "Existing Pickle Model better than Newly Trained Model..."
                                )
                        print("-" * 100)

                        print("\n")
                        print("-" * 100)
                        print("Inference Model Information")
                        print("-" * 100)
                        with open(rf"{model_filename}", "rb") as f:
                            data_from_pickle = pickle.load(f)
                        data_from_pickle_model = data_from_pickle["model"]
                        data_from_pickle_largest_mean_score = data_from_pickle[
                            "largest_mean_score"
                        ]
                        data_from_pickle_train_data_size = data_from_pickle[
                            "train_data_size"
                        ]
                        data_from_pickle_date_created = data_from_pickle["date_created"]

                        print(f"Inference Model Type: {data_from_pickle_model}")
                        print(
                            f"Largest Mean Score: {data_from_pickle_largest_mean_score}"
                        )
                        print(
                            f"Train Data Size Used: {data_from_pickle_train_data_size}"
                        )
                        print(f"Date Created: {data_from_pickle_date_created}")
                        print("-" * 100)

                        end_time = time.time()

                        print("\n")
                        print(
                            f"Total time to complete evaluation: {(end_time - start_time):.2f} seconds"
                        )

                    except Exception as e:
                        print(f"An error occured: {e}")
        except Exception as e:
            logging.error(
                f"An error occured when evaluating the model population and returning the best performing model: {e}"
            )


class Inference:

    @staticmethod
    def load_model_for_inference(filename: str) -> object:
        logging.info(f"Loading saved model {filename} for inference...")
        try:
            with open(rf"{filename}", "rb") as f:
                data_from_pickle = pickle.load(f)
                loaded_model = data_from_pickle["model"]
            logging.info(f"...Loaded saved model {filename} for inference.")
        except Exception as e:
            logging.error(
                f"An error occured when attempting to load saved model {filename} for inference: {e}"
            )

        return loaded_model

    @staticmethod
    def perform_inference_using_loaded_model(
        model: object, preprocessed_data: pandas.DataFrame
    ) -> pandas.DataFrame:
        logging.info("Performing inference using loaded model...")
        try:
            predictions = model.predict(preprocessed_data)
            predictions_dataframe = pandas.DataFrame({"predicitons": predictions})
            logging.info("...Inference using loaded model complete.")
        except Exception as e:
            logging.error(
                f"An error occured when performing inference using loaded model: {e}"
            )

        return predictions_dataframe


if __name__ == "__main__":
    logging.info("-" * 100)
    logging.info("Starting Mechanicus Training Pipeline")
    logging.info("-" * 100)
    CV = 5
    SCORING_METRIC = "accuracy"
    start_time = time.time()

    response_variable_production = (
        "position_hash"  
    )


    training_data = DataCollector.load_servo_eeg_data("training_data.json")

    ExploratoryDataAnalysis.get_summary_statistics(
        training_data, filename="training_data_eda.txt"
    )

    preprocessed_data = PreprocessData.preprocess_data(
        training_data, response_variable_production, Phase("training")
    )
    x = preprocessed_data[0]
    y = preprocessed_data[1]

    TrainModel.evaluateModel(x=x, y=y, cv=CV, scoring_metric=SCORING_METRIC)

    inference_model = Inference.load_model_for_inference(filename="inference_model.pkl")

    inference_data = DataCollector.load_servo_eeg_data("inference_data.json")
    ExploratoryDataAnalysis.get_summary_statistics(
        inference_data, filename="inference_data_eda.txt"
    )

    preprocessed_inference_data = PreprocessData.preprocess_data(
        inference_data, response_variable_production, Phase("inference")
    )
    preprocessed_inference_x = preprocessed_inference_data[0]

    predictions = Inference.perform_inference_using_loaded_model(
        model=inference_model, preprocessed_data=preprocessed_inference_x
    )
    predictions_results_path = "predictions_output.csv"
    logging.info(
        f"Writing predictions results to output file: {predictions_results_path}..."
    )
    predictions.to_csv(predictions_results_path, mode="w", index=False)

    end_time = time.time()

    logging.info(
        f"----------COMPLETED Mechanicus Training Pipeline in {end_time-start_time} seconds---------"
    )
