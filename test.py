# Data Preparation
import os
import shutil
from PIL import Image
import numpy as np
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator

# Prediction Image Lib
from keras.preprocessing import image
from google.colab import files
from tensorflow.keras.applications.xception import preprocess_input
from google.api_core.client_options import ClientOptions

PROJECT_ID = os.environ['tomatreat']
VERSION_NAME = os.environ['v001']
MODEL_NAME = os.environ['tomatreat_model_1']

labels = list(train_generator.class_indices.keys())

#uploaded = files.upload()
uploaded = Image.open('leaf-tomato.jpg')

for fn in uploaded.keys():
  path = fn
  img = image.load_img(path, target_size=(256, 256))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  images = np.vstack([x])
  proba = best_model.predict(images)[0]
  plt.imshow(img)
  plt.show()
  print(proba)
  for i in range(len(labels)):
    print("{}: {:.2f}%".format(labels[i], proba[i]* 100))