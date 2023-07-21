import pandas as pd
import json
from src.constants import *
from pymongo.mongo_client import MongoClient

# Create a new client and connect to the server
client = MongoClient(MONGO_URI)

df = pd.read_csv("./notebook/wafer_23012020_041211.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
df.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
json_records = list(json.loads(df.T.to_json()).values())

# dump data into data base
client[MONGO_DB_NAME][MONGO_COLLECTION_NAME].insert_many(json_records)

