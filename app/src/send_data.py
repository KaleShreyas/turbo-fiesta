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
print("Total data : ", X_test.shape)

ndarray = X_test.toarray()
data = ndarray.tolist()

# Taking only few rows for quick run
data = data[:25]
y_test = y_test[:25]

with open("target.csv", 'w') as f_target:
    for idx, row in enumerate(data):
        print("Row # ", idx)
        send_row = dict(enumerate(row, start=0))
        print("Input :", send_row)
        print("Actual : ", y_test[idx])
        f_target.write(f"{X_test[idx]}\n")
        resp = requests.post("http://127.0.0.1:9696/predict",
                            headers={"Content-Type": "application/json"},
                            data=json.dumps(send_row)).json()
        print(f"Prediction: {resp['renewable-prediction']}")
        sleep(1)
