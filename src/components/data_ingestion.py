import sys
import os

import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path

from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(ARTIFACT_FOLDER)


class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name) -> pd.DataFrame:
        """
            Method Name : export_collection_as_dataframe
            Description : This methods fetch data from the MongoDB

            Output : Returns the fetched data in form of pd.DataFrame
            On Failure  :   Write an exception log and then raise an exception

            Version : 0.1
        """

        try:
            client = MongoClient(MONGO_URI)

            collection = client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.tolist():
                df = df.drop(columns=['_id'])

            df.replace({"na": np.nan}, inplace=True)
            df.rename({"Good/Bad": TARGET_COLUMN}, inplace=True)
            # print("df=>",df)
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store_file(self) -> Path:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method reads data from mongodb and saves it into artifacts. 

        Output      :   dataset is returned as a pd.DataFrame
        On Failure  :   Write an exception log and then raise an exception

        Version     :   0.1 
        """

        try:
            logging.info("Exporting data from the MongoDb")
            raw_file_path = self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path, exist_ok=True)

            sensor_data = self.export_collection_as_dataframe(
                db_name=MONGO_DB_NAME,
                collection_name=MONGO_COLLECTION_NAME
            )
            logging.info(
                f"Saving exported data to raw file path : {raw_file_path}")
            feature_store_file_path = os.path.join(
                raw_file_path,
                "wafer_data.csv"
            )

            sensor_data.to_csv(feature_store_file_path, index=False)

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def initate_data_ingestion(self) -> Path:
        """
            Method Name :   initiate_data_ingestion
            Description :   This method initiates the data ingestion components of training pipeline 

            Output      :   Path of train set and test set are returned as the artifacts of data ingestion components
            On Failure  :   Write an exception log and then raise an exception

            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        logging.info(
            "Entered intiate_data_ingestion process of DataIngestion class")
        try:
            feature_store_file_path = self.export_data_into_feature_store_file()

            logging.info("Got the data from MongoDB")

            logging.info("Exiting intiate_data_ingestion process")

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys) from e
