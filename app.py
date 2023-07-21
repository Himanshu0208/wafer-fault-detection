import os
import sys
from flask import Flask, render_template, jsonify, send_file, request

from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.pipeline.predict_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainingPipeline

app = Flask(__name__)


@app.route("/")
def home():
    try:
        message = {
            "Curent_Page": "Home",
            "Message": "Welcome to the API",
            "Functions_Available": {
                "/train": "For training",
                "/predict": "For giving data to predict"
            }
        }
        return jsonify(message)
    except Exception as e:
        raise CustomException(e, sys)


@app.route("/train")
def train():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        message = {
            "Curent_Page": "Training Model",
            "Message": "Training page of the model is completed Successfully",
            "Functions_Available": {
                "/train": "For training",
                "/predict": "For giving data to predict"
            }
        }
        return jsonify(message)

    except Exception as e:
        raise CustomException(e, sys)


@app.route("/predict", methods=["POST", "GET"])
def upload():
    try:
        if request.method == "POST":
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_details = prediction_pipeline.run_pipeline()

            logging.info("Prediction Done")
            return send_file(
                path_or_file=prediction_file_details.prediction_file_path,
                download_name=prediction_file_details.prediction_file_name,
                as_attachment=True
            )
        else:
            return render_template("file_upload.html")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
