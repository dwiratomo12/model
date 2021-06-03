from keras.preprocessing import image
from google.colab import files
from tensorflow.keras.applications.xception import preprocess_input

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

MODEL_URI = 'http://localhost:5000/v1/models/daun:predict'

# lokasi di vmnya
model_path = "./deploy_model_daun_atau_bukan"  # di sini letak model ml nya
ext_model = tf.keras.models.load_model(model_path)

model_path = "/content/drive/MyDrive/Colab Notebooks/model.hdf5"
classes = ['daun', 'nondaun']
# Load the model
best_model = load_model(model_path, compile=True)

#labels = list(train_generator.class_indices.keys())
labels = list('labels.txt')

print(labels)
uploaded = files.upload()
daun = 0
nondaun = 0

for fn in uploaded.keys():
    path = fn
    img = image.load_img(path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    images = np.vstack([x])
    proba = best_model.predict(images)[0]
    classes = best_model.predict(images, batch_size=10)
    # plt.imshow(img)
    # plt.show()
    print(classes[0])
    if classes[0] < 1:
        print(fn + " daun")
        daun = daun + 1

    else:
        print(fn + " nondaun")
        nondaun = nondaun + 1

print(daun)
print(nondaun)
