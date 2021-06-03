from flask import Flask, request
import os
import json

import numpy as np
import requests
from PIL import Image
from keras.preprocessing import image
from prediksilabel import predict as prediction_label
# from prediksi_daun import predict as prediction_daun

# Data Preparation
import os
import shutil
from PIL import Image
import numpy as np
import cv2
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator
import json
import matplotlib.pyplot as plt
import tensorflow as tf

# Prediction Image Lib
from keras.preprocessing import image
from google.colab import files
from tensorflow.keras.applications.xception import preprocess_input
from google.api_core.client_options import ClientOptions

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
    # lokasi di vmnya
    model_path = "./deploy_model_1"  # di sini letak model ml nya
    ext_model = tf.keras.models.load_model(model_path)

    # labels = list(train_generator.class_indices.keys())
    labels = list('labels.txt')

    #uploaded = files.upload()
    uploaded = image.load_img('images/leaf-tomato.jpg')

    for fn in uploaded.keys():
        path = fn
        img = image.load_img(path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        images = np.vstack([x])
        # proba = best_model.predict(images)[0]
        proba = ext_model.predict(images)[0]
        plt.imshow(img)
        plt.show()
        print(proba)
        tertinggi = max(proba)
        for i in range(len(labels)):
            print("{}: {:.2f}%".format(labels[i], proba[i] * 100))
            if proba[i] == tertinggi:
                fix_label = i
    # prediction_string = [str(pred) for pred in proba]
    response_json = {
        "data": img,
        "label": str(labels[fix_label]),
        # "data" : data.get("instances"),
        "prediction": str(proba[fix_label] * 100)
    }

    # response_json = prediction_label()

    # return response_json
    return json.dumps(response_json)


if __name__ == '__main__':
    # app.run()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
