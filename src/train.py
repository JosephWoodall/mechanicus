import pandas
import numpy
import glob
import os
import random
from contextlib import redirect_stdout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel

import xgboost
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

    def generate_offline_eeg_data(
        rows: int = 10000, cols: int = 9759
    ) -> pandas.DataFrame:
        """Creates a pandas DataFrame similar to offline EEG data with random float values between -1 and 1.

        Args:
            rows (int, optional): number of rows for the ouput. Defaults to 64.
            cols (int, optional): number of columns for the output. Defaults to 9759.

        Returns:
            pandas.DataFrame: pandas DataFrame similar to offline EEG data with random float values between -1 and 1.
        """

        data = numpy.random.uniform(low=-1, high=1, size=(rows, cols))

        column_names = [f"{i+1}" for i in range(cols)]

        return_data = pandas.DataFrame(data, columns=column_names)

        return_data["activity_type"] = random.choice(
            ["baseline_eyes_open", "baseline_eyes_closed"]
        )

        return return_data

    def collect_offline_eeg_data(data_dir: str = None) -> pandas.DataFrame:
        """This method collects data from https://www.physionet.org/content/eegmmidb/1.0.0/S001/#files-panel. A summary of the data is shown below:

        Abstract
        This data set consists of over 1500 one- and two-minute EEG recordings, obtained from 109 volunteers, as described below.

        Experimental Protocol
        Subjects performed different motor/imagery tasks while 64-channel EEG were recorded using the BCI2000 system (http://www.bci2000.org). Each subject performed 14 experimental runs: two one-minute baseline runs (one with eyes open, one with eyes closed), and three two-minute runs of each of the four following tasks:

        - A target appears on either the left or the right side of the screen. The subject opens and closes the corresponding fist until the target disappears. Then the subject relaxes.
        - A target appears on either the left or the right side of the screen. The subject imagines opening and closing the corresponding fist until the target disappears. Then the subject relaxes.
        - A target appears on either the top or the bottom of the screen. The subject opens and closes either both fists (if the target is on top) or both feet (if the target is on the bottom) until the target disappears. Then the subject relaxes.
        A-  target appears on either the top or the bottom of the screen. The subject imagines opening and closing either both fists (if the target is on top) or both feet (if the target is on the bottom) until the target disappears. Then the subject relaxes.
        In summary, the experimental runs were:

        1. Baseline, eyes open
        2. Baseline, eyes closed
        3. Task 1 (open and close left or right fist)
        4. Task 2 (imagine opening and closing left or right fist)
        5. Task 3 (open and close both fists or both feet)
        6. Task 4 (imagine opening and closing both fists or both feet)
        7. Task 1
        8. Task 2
        9. Task 3
        10. Task 4
        11. Task 1
        12. Task 2
        13. Task 3
        14. Task 4
        The data are provided here in EDF+ format (containing 64 EEG signals, each sampled at 160 samples per second, and an annotation channel). For use with PhysioToolkit software, rdedfann generated a separate PhysioBank-compatible annotation file (with the suffix .event) for each recording. The .event files and the annotation channels in the corresponding .edf files contain identical data.

        Each annotation includes one of three codes (T0, T1, or T2):

        T0 corresponds to rest
        T1 corresponds to onset of motion (real or imagined) of
        the left fist (in runs 3, 4, 7, 8, 11, and 12)
        both fists (in runs 5, 6, 9, 10, 13, and 14)
        T2 corresponds to onset of motion (real or imagined) of
        the right fist (in runs 3, 4, 7, 8, 11, and 12)
        both feet (in runs 5, 6, 9, 10, 13, and 14)
        In the BCI2000-format versions of these files, which may be available from the contributors of this data set, these annotations are encoded as values of 0, 1, or 2 in the TargetCode state variable.

        Montage
        The EEGs were recorded from 64 electrodes as per the international 10-10 system (excluding electrodes Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, and P10), as shown in this PDF figure. The numbers below each electrode name indicate the order in which they appear in the records; note that signals in the records are numbered from 0 to 63, while the numbers in the figure range from 1 to 64.

        Acknowledgments
        This data set was created and contributed to PhysioBank by Gerwin Schalk (schalk at wadsworth dot org) and his colleagues at the BCI R&D Program, Wadsworth Center, New York State Department of Health, Albany, NY. W.A. Sarnacki collected the data. Aditya Joshi compiled the dataset and prepared the documentation. D.J. McFarland and J.R. Wolpaw were responsible for experimental design and project oversight, respectively. This work was supported by grants from NIH/NIBIB ((EB006356 (GS) and EB00856 (JRW and GS)).



        Returns:
            pandas.DataFrame: instance of the data collected offline.
        """
        import mne

        filenames = glob.glob(os.path.join(data_dir, "**", "*.edf"), recursive=True)

        dataframes = []

        for i in filenames[:3]:
            activity_type_in_i = i[i.find("R") + 1 : i.find(".edf")]

            eeg_data = mne.io.read_raw_edf(f"{i}").get_data()

            eeg_data = pandas.DataFrame(eeg_data)

            if activity_type_in_i == "01":
                eeg_data["activity_type"] = "baseline_eyes_open"
            elif activity_type_in_i == "02":
                eeg_data["activity_type"] = "baseline_eyes_closed"
            elif activity_type_in_i == "03":
                eeg_data["activity_type"] = "task_1_open_and_close_left_or_right_fist"
            elif activity_type_in_i == "04":
                eeg_data["activity_type"] = (
                    "task_2_imagine_opening_and_closing_left_or_right_fist"
                )
            elif activity_type_in_i == "05":
                eeg_data["activity_type"] = (
                    "task_3_open_and_close_both_fists_or_both_feet"
                )
            elif activity_type_in_i == "06":
                eeg_data["activity_type"] = (
                    "task_4_imagine_opening_and_closing_both_fists_or_feet"
                )
            elif activity_type_in_i == "07":
                eeg_data["activity_type"] = "task_1"
            elif activity_type_in_i == "08":
                eeg_data["activity_type"] = "task_2"
            elif activity_type_in_i == "09":
                eeg_data["activity_type"] = "task_3"
            elif activity_type_in_i == "10":
                eeg_data["activity_type"] = "task_4"
            elif activity_type_in_i == "11":
                eeg_data["activity_type"] = "task_1"
            elif activity_type_in_i == "12":
                eeg_data["activity_type"] = "task_2"
            elif activity_type_in_i == "13":
                eeg_data["activity_type"] = "task_3"
            elif activity_type_in_i == "14":
                eeg_data["activity_type"] = "task_4"
            else:
                eeg_data["activity_type"] = "none_found_from_source_file"

            eeg_data["subject_id"] = os.path.basename(i).split(".")[0].split("R")[0]

            dataframes.append(eeg_data)
        output_eeg_data = pandas.concat(dataframes, ignore_index=True)

        return output_eeg_data

    def generate_offline_eeg_data_with_cartesian_plane_position_of_movement(
        rows: int = 10000, cols: int = 5
    ) -> pandas.DataFrame:
        """Generates synthetic eeg data which matches to a 3D cartesian plane, with mean of 0 (pointed at zenith) of prosthetic movement.
        The idea is that upon each movement, eeg data corresponds to the angle of which the prosthetic will point. So at 0 (zenith), the
        prosthetic will be directionally parallel to the limb it is upon. In other words, the prosthetic will be straight.
        Ultimately, this is what the data will look like, so it is being generated here. All 3d_position values will be greater than 0 because
        the prosthetic cannot physically invert upon itself.

        The intuition is that I am able to predict what the 3D position of the prosthetic is based on the corresponding eeg data.

        TODO: FIGURE OUT HOW TO CREATE A UNIQUE MAPPING OF THE prosthetic_cartesian_3d_position 3D MAPPING OUTPUT TO A SINGLE, UNIQUE NUMBER CORRESPONDING TO EACH OF THE TUPLE VALUES

        Args:
            rows (int, optional): Number of rows in the DataFrame. Defaults to 500.
            cols (int, optional): Number of columns in the DataFrame. Defaults to 9999.

        Returns:
            pandas.DataFrame: _description_
        """
        # Generate features matrix with incremental values
        base_features = numpy.arange(rows * cols).reshape(rows, cols)

        # Center and normalize features to mean=0, std=1
        features_mean = base_features.mean()
        features_std = base_features.std(
            ddof=1
        )  # Using ddof=1 for sample standard deviation
        features_normalized = (base_features - features_mean) / features_std

        # Create DataFrame with features
        df = pandas.DataFrame(
            features_normalized, columns=[f"eeg_{i}" for i in range(cols)]
        )

        # Generate response values (3-tuples with mean=0, std=1)
        response_base = numpy.random.normal(loc=0, scale=1, size=(rows, 3))
        response_mean = response_base.mean(axis=1, keepdims=True)
        response_std = response_base.std(axis=1, keepdims=True)
        response_normalized = (response_base - response_mean) / response_std

        # Ensure all values are >= 0 while preserving mean=0, std=1
        response_positive = numpy.abs(response_normalized)
        response_mean_final = response_positive.mean(axis=1, keepdims=True)
        response_std_final = response_positive.std(axis=1, keepdims=True)
        response_final = (response_positive - response_mean_final) / response_std_final

        # Add response column as tuples
        df["prosthetic_cartesian_3d_position"] = [tuple(row) for row in response_final]
        df["prosthetic_cartesian_3d_position_hash_value"] = df[
            "prosthetic_cartesian_3d_position"
        ].apply(lambda x: abs(hash(",".join(f"{v:.10f}" for v in x))))

        return df


class ExploratoryDataAnalysis:

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
                            # (
                            #    "XGBoost Classifier",
                            #    xgboost.XGBClassifier(booster="gblinear"),
                            # ),
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
                            # "XGBoost Classifier": {
                            # "n_estimators": generateRandomIntList(),
                            # "max_depth": generateRandomIntList(),
                            # "max_leaves": generateRandomIntList(),
                            # "max_bin": generateRandomIntList(),
                            # "grow_policy": ["depthwise", "lossguide"],
                            # "learning_rate": generateRandomIntList(),
                            # "gamma": generateRandomIntList(),
                            # "min_child_weight": generateRandomFloatList(),
                            # "subsample": generateRandomFloatList(),
                            # "sampling_method": ["uniform", "gradient_based"],
                            # "colsample_bytree": generateRandomFloatList(),
                            # "colsample_bylevel": generateRandomFloatList(),
                            # "colsample_bynode": generateRandomFloatList(),
                            # "reg_alpha": generateRandomFloatList(),
                            # "reg_lambda": generateRandomFloatList(),
                            # "scale_pos_weight": generateRandomFloatList(),
                            # "max_features": ["auto", "sqrt", "log2"],
                            # },
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
    CV = 10
    SCORING_METRIC = "accuracy"
    start_time = time.time()

    response_variable_production = (
        "activity_type"  # "prosthetic_cartesian_3d_position_hash_value"
    )

    # training_data = DataCollector.collect_offline_eeg_data(
    #    data_dir="src/data/eeg-motor-movementimagery-dataset-1.0.0/training"
    # )
    training_data = DataCollector.generate_offline_eeg_data()

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
    # inference_data = DataCollector.collect_offline_eeg_data(
    #    data_dir="src/data/eeg-motor-movementimagery-dataset-1.0.0/inference"
    # )
    inference_data = DataCollector.generate_offline_eeg_data()
    ExploratoryDataAnalysis.get_summary_statistics(
        inference_data, filename="inference_data.txt"
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
