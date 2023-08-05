import os
import json
import pickle
from datetime import datetime
from time import sleep
import pyarrow as pa


import pandas as pd
import requests

# table = pd.read_csv("./evidently_service/datasets/test.csv")
# table = pa.Table.from_pandas(table, preserve_index=True)
# data  = table.to_pylist()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
data_path = "./output"
X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

ndarray = X_test.toarray()
data = ndarray.tolist()

with open("target.csv", 'w') as f_target:
    for row in data:
        send_row = dict(enumerate(row, start=0))
        print(send_row)
        # f_target.write(f"{row['0']}\n")
        resp = requests.post("http://127.0.0.1:9696/predict",
                            headers={"Content-Type": "application/json"},
                            data=json.dumps(send_row)).json()
        print(f"prediction: {resp['renewable-prediction']}")
        sleep(1)
