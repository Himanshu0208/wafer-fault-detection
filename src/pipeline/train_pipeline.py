import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException


class TrainerPipeline:
  def __init__(self) -> None:
    self.data_ingestion = DataIngestion()
    self.data_transformation = DataTransformation()
    self.model_trainer = ModelTrainer()

  


