import sys
import numpy as np
from zipfile import Path
from typing import Tuple

from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.main_utils import MainUtils
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def start_data_ingestion(self) -> Path:
        try:
            data_ingestion = DataIngestion()
            feature_store_path = data_ingestion.initate_data_ingestion()
            return feature_store_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, file_path: str) -> Tuple[np. array, np.array, Path]:
        try:
            data_transformation = DataTransformation(file_path)
            (
                train_arr,
                test_arr,
                preprocessor_path) = data_transformation.initiate_data_transformation()

            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_arr: np.array, test_arr: np.array) -> float:
        try:
            model_trainer = ModelTrainer()
            best_model_score = model_trainer.initiate_model_trainer(
                train_arr=train_arr,
                test_arr=test_arr)

            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> None:
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(
                feature_store_file_path
            )
            accuracy = self.start_model_training(
                train_arr=train_arr,
                test_arr=test_arr
            )

            logging.info(
                f"Training Completed,  Trained model score : {accuracy}"
            )

        except Exception as e:
            raise CustomException(e, sys)
