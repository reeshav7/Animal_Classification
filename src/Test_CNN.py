import os
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = r"models/cnn_animals_best.h5"
TEST_IMG_DIR = r"../test_images"

IMG_SIZE = 224

class_names = ['butterfly','cat','chicken','cow','dog',
               'elephant','horse','sheep','spider','squirrel']

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully ✔")

print("\n========== CNN BATCH TEST ==========\n")

for img_name in os.listdir(TEST_IMG_DIR):
    if img_name.lower().endswith((".jpg",".jpeg",".png",".bmp",".jfif",".webp")):

        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        arr = np.array(img)/255.0
        arr = np.expand_dims(arr,0)

        pred = model.predict(arr,verbose=0)[0]
        idx = np.argmax(pred)
        conf = pred[idx]*100

        print(f"{img_name:<25} --> {class_names[idx].upper():<10} ({conf:.2f}%)")

print("\n====================================\n")
