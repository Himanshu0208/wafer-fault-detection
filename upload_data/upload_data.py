import pandas as pd
import json
from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://himanshupandey1036:kHia1RHFwJIMMjDX@cluster0.z0nfsde.mongodb.net/?retryWrites=true&w=majority"
DATABASE_NAME = "waferfault"
COLLECTION_NAME = "sensor_data"

# Create a new client and connect to the server
client = MongoClient(uri)

df = pd.read_csv("./notebook/wafer_23012020_041211.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
json_records = list(json.loads(df.T.to_json()).values())

# dump data into data base
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)

