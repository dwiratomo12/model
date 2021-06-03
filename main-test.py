from flask import Flask, request
import os
import json

import numpy as np
from PIL import Image
from keras.preprocessing import image
from prediksi_label import predict as prediction_label
# from prediksi_daun import predict as prediction_daun

app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    data = {"status": 200, "data": "Hello world"}
    umur = {
        "budi": 20,
        "rani": 21,
        "rara": 23,
        "ruru": 19
    }
    data["val_umur"] = umur
    return json.dumps(data)


@app.route('/api/predict/test', methods=['GET'])
def predict_test():
    # request_json = request.json
    # print("data: {}".format(request_json))
    # print("type: {}".format(type(request_json)))

    uploaded = Image.open('images/leaf-tomato.jpg')

    response_json = prediction_label(uploaded.tolist())

    return response_json


if __name__ == '__main__':
    # app.run()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
