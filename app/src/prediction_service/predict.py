import os
import pickle

import requests
from flask import Flask
from flask import request
from flask import jsonify

from pymongo import MongoClient


MODEL_FILE = os.getenv('MODEL_FILE', 'model.pkl')

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

with open(MODEL_FILE, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


app = Flask('renewable-prediction')
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

def make_encoding(object):
    if list(object.items())[0][1] == "Male":
        object["Gender"] = "1"
    else:
        object["Gender"] = "0"
    return object

@app.route('/predict', methods=['POST'])
def predict():
    object = request.get_json()
    object_enc = make_encoding(object)

    object_dv = dv.transform(object_enc)

    pred = model.predict(object_dv)
    result = {
        'renewable-prediction': float(pred)
    }

    save_to_db(object, float(pred))
    send_to_evidently_service(object, float(pred))
    return jsonify(result)


def save_to_db(object, pred):
    obj = object.copy()
    obj['prediction'] = pred
    collection.insert_one(dict(obj))


def send_to_evidently_service(object, pred):
    obj = object.copy()
    obj['prediction'] = pred
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/renewable", json=[obj])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)