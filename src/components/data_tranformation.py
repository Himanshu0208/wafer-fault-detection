import os
import sys
import pandas as pd
import numpy as np
from zipfile import Path
from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler, FunctionTransformer

from typing import Tuple

from src.exception import CustomException
from src.logger import logging
from src.constants import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    artifacts_dir: str = os.path.join(ARTIFACT_FOLDER)
    transformed_test_data_dir = os.path.join(artifacts_dir, "test")
    transformed_test_data_file = os.path.join(
        transformed_test_data_dir, "test.csv")
    transformed_object_dir = os.path.join(artifacts_dir, "preprocessor")
    transformed_object_file_path = os.path.join(
        transformed_object_dir,
        f"{PREPROCESSOR_FILE_NAME}{PREPROCESSOR_FILE_EXTENSION}"
    )


class DataTransformation:
    def __init__(self, feature_store_file_path) -> None:
        self.utils = MainUtils()
        self.data_transformation_config = DataTransformationConfig()
        self.feature_store_file_path = feature_store_file_path

        os.makedirs(
            self.data_transformation_config.transformed_object_dir, exist_ok=True)
        os.makedirs(
            self.data_transformation_config.transformed_test_data_dir, exist_ok=True)

    # @staticmethod
    def get_data(self, feature_store_file_path: str) -> pd.DataFrame:
        """
        Method Name :   get_data
        Description :   This method reads all the validated raw data from the feature_store_file_path and returns a pandas DataFrame containing the merged data. 

        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            df = pd.read_csv(feature_store_file_path)
            df.rename({"Good/Bad": TARGET_COLUMN})

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:

        try:
            preprocessor = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
                    ('scaler', RobustScaler()),
                    ('smt', SMOTETomek(sampling_strategy='auto')),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> Tuple[np.array, np.array, Path]:
        """
            Method Name :   initiate_data_transformation
            Description :   This method initiates the data transformation component for the pipeline 

            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception

            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        logging.info(
            "Entered initiate_data_transformation method of DataTransformation Class")
        try:
            df = self.get_data(
                feature_store_file_path=self.feature_store_file_path
            )

            X = df.drop(columns=[TARGET_COLUMN])
            y = np.where(df[TARGET_COLUMN] == -1, 0, 1)  # replacing -1 to 0

            preprocessor = self.get_data_transformer_object()
            # X_res, y_res = preprocessor[-1].fit_resample(X,y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train_scaled,y_train = preprocessor.fit_resample(X_train, y_train)
            X_test_scaled = preprocessor[:-1].transform(X_test)
            
            # Saving data for testing
            df_test = pd.DataFrame(data=X_test, columns=list(X.columns))
            df_test.to_csv(
                self.data_transformation_config.transformed_test_data_file
            )


            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            self.utils.save_object(
                file_path=preprocessor_path,
                object_to_save=preprocessor
            )

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            return (train_arr, test_arr, preprocessor_path)

        except Exception as e:
            raise CustomException(e, sys)
