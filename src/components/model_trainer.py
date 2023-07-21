import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from typing import Tuple

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

from src.constants import *
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    articat_folder = os.path.join(ARTIFACT_FOLDER)
    trained_model_path = os.path.join(
        articat_folder, 
        f"{MODEL_FILE_NAME}{MODEL_FILE_EXTENSION}"
        )
    expected_accuracy = 0.60
    model_config_file_path = os.path.join('config', 'model.yaml')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.model = {
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'LGBMClassifier': LGBMClassifier(),
        }

    def evaluate_models(self, x_train: np.array, y_train: np.array, models: object) -> dict:
        try:
            X_train, Y_train, X_test, Y_test = train_test_split(
                x_train,
                y_train,
                train_size=0.2,
                random_state=42
            )

            report = {}
            models_values = list(models.values())
            models_keys = list(models.keys())
            for i in range(len(models_values)):
                model = models_values[i]

                # fitting the model
                model.fit(X_train, Y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # getting the score of the model
                trained_model_score = accuracy_score(
                    y_pred=y_train_pred, y_true=y_train)
                test_model_score = accuracy_score(
                    y_pred=y_test_pred, y_true=y_test_pred)

                report[models_keys[i]] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(self, x_train: np.array, y_train: np.array) -> Tuple[str, object, float]:
        try:

            # evaluate all models
            model_report: dict = self.evaluate_models(
                x_train=x_train,
                y_train=y_train,
                models=self.model
            )

            # Getting the best model score
            best_model_score = max(list(model_report.values()))

            # Getting the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Extracting the best model
            best_model = self.model[best_model_name]

            return (best_model_name, best_model, best_model_score)
        except Exception as e:
            raise CustomException(e, sys)

    def fine_tune_model(
        self,
        best_model_name: str,
        best_model_object: object,
        x_train: np.array,
        y_train: np.array
    ) -> object:

        try:
            model_param_grid = self.utils.read_yaml_file(
                file_path=self.model_trainer_config.model_config_file_path
            )["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid = GridSearchCV(
                estimator=best_model_object,
                param_grid=model_param_grid,
                cv=10,
                n_jobs=-1,
                verbose=1
            )

            grid.fit(x=x_train, y=y_train)

            best_params = grid.best_params_

            logging.info(
                f"The best params for : '{best_model_name}' are {best_params}")

            finned_tuned_model = best_model_object.set_params(**best_params)

            return finned_tuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_arr, test_arr) -> float:
        logging.info("Entered initiate_model_trainer method of ModelTrainer")
        try:

            logging.info("Extracting data from train and test array")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            logging.info("getting the best model")
            best_model_name, best_model, best_model_score = self.get_best_model(
                x_train=x_train,
                y_train=y_train
            )

            # Fine tuning the model
            best_model = self.fine_tune_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                x_train=x_train,
                y_train=y_train
            )

            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_true=y_test, y_pred=y_pred)
            logging.info(
                f"Best Model is : {best_model} , with score : {best_model_score}"
            )

            if best_model_score < self.model_trainer_config.expected_accuracy:
                logging.error(
                    "The best model accuracy is less than the excepcted threshold i.e 0.6")
                raise Exception(
                    "The best model accuracy is less than the excepcted threshold i.e 0.6")

            logging.info(
                f"Saving object model at {self.model_trainer_config.trained_model_path}"
            )

            os.makedirs(
                self.model_trainer_config.trained_model_path, exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                object_to_save=best_model
            )

            logging.info(
                "Exiting initiate_model_trainer method in ModelTrainer"
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
