import os
import sys
import pandas as pd
import numpy as np

from zipfile import Path
from typing import Tuple, List, Set, Dict

from flask import request
from datetime import datetime

from src.utils.main_utils import MainUtils
from src.exception import CustomException
from src.logger import logging
from src.constants import *
from dataclasses import dataclass


@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = f"prediction-{datetime.now()}.csv"

    model_file_path = os.path.join(
        ARTIFACT_FOLDER,
        f"{MODEL_FILE_NAME}",
        f"{MODEL_FILE_NAME}{MODEL_FILE_EXTENSION}"
    )
    preprocessor_file_path = os.path.join(
        ARTIFACT_FOLDER,
        f"{PREPROCESSOR_FILE_NAME}",
        f"{PREPROCESSOR_FILE_NAME}{PREPROCESSOR_FILE_EXTENSION}"
    )
    prediction_file_path = os.path.join(
        prediction_output_dirname,
        prediction_file_name
    )


class PredictionPipeline:
    def __init__(self, incoming_request: request) -> None:
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.utils = MainUtils()
        self.request = incoming_request

    def save_input_files(self) -> Path:
        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 

            Output      :   Saved input dataframe file path 
            On Failure  :   Write an exception log and then raise an exception

            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            pred_file_dir_name = PREDICTION_ARTIFACT_FOLDER
            os.makedirs(pred_file_dir_name, exist_ok=True)

            input_csv = request.files['file']
            prediction_data_file_path = os.path.join(
                pred_file_dir_name,
                input_csv.filename
            )

            input_csv.save(prediction_data_file_path)

            return prediction_data_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> np.array:
        try:
            model: object = self.utils.load_obj(
                file_path=self.prediction_pipeline_config.model_file_path
            )
            preprocessor = self.utils.load_obj(
                file_path=self.prediction_pipeline_config.preprocessor_file_path
            )

            transformed_features = preprocessor[:-1].transform(features)
            y_pred = model.predict(transformed_features)

            return y_pred

        except Exception as e:
            raise CustomException(e, sys)

    def get_predicted_dataframe(self, file_path) -> None:
        """
            Method Name :   get_predicted_dataframe
            Description :   this method returns the dataframw with a new column containing predictions

            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception

            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            prediction_column_name: str = TARGET_COLUMN
            input_dataframe = pd.read_csv(file_path)
            input_dataframe = input_dataframe.drop(
                columns=["Unnamed: 0"]) if "Unnamed: 0" in input_dataframe.columns else input_dataframe

            predictions = self.predict(input_dataframe)

            input_dataframe[prediction_column_name] = [
                pred for pred in predictions]

            target_column_mapping = {0: 'bad', 1: 'good'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(
                target_column_mapping
            )

            os.makedirs(
                name=self.prediction_pipeline_config.prediction_output_dirname,
                exist_ok=True)
            input_dataframe.to_csv(
                self.prediction_pipeline_config.prediction_file_path)
            logging.info("Prediction completed")

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            input_data_file_path = self.save_input_files()
            self.get_predicted_dataframe(input_data_file_path)
            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(e, sys)
