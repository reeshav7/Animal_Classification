import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Loading Trained Model

model = tf.keras.models.load_model("../models/cnn_animals_mobilenet.h5")

test_dir = "../test_images"
img_size = (224,224)
class_names = list(model.class_names) if hasattr(model,"class_names") else ["cats","dogs"]

print("\n========== CNN Batch Test Result ==========\n")

for file in os.listdir(test_dir):
    if file.lower().endswith((".jpg",".jpeg",".png",".jfif",".bmp",".webp")):

        path = os.path.join(test_dir,file)

        img = load_img(path, target_size=img_size)
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0]
        idx = np.argmax(pred)
        label = class_names[idx]
        conf = pred[idx] * 100

        print(f"{file:<18} --> {label.upper()}  ({conf:.2f}%)")

print("\n===========================================\n")
