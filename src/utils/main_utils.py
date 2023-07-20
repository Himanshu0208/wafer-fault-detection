import os
import sys
import pandas as pd
import numpy as np
import pickle
import yaml

from zipfile import Path
from src.exception import CustomException
from src.logger import logging


class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, file_path : str) -> dict :
        try :
            with open(file=file_path, mode="rb") as yaml_file:
                return yaml.safe_load(yaml_file)
            
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def save_object(file_path: str, object_to_save: object) -> None :
        logging.info("Entered save_object method of the MainUtils class")
        try :
            with open(file_path , 'wb') as file_obj:
                pickle.dump(obj=object_to_save , file=file_obj)

            logging.info("Exited the save_object method of the MainUtils class")

        except Exception as e:
            raise CustomException(e,sys)
