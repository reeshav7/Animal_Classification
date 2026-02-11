import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# ========== Paths ==========
MODEL_PATH = "models/ann_model.h5"
SCALER_PATH = "models/scaler.pkl"
PCA_PATH = "models/pca.pkl"
TEST_DIR = "../test_images"

IMG_SIZE = 64

# ========== Load Model ==========
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

class_names = ["cat", "dog"]

print("\n================ ANN Batch Testing ================\n")

for file in os.listdir(TEST_DIR):
    if file.lower().endswith(("jpg", "jpeg", "png", "jfif")):

        img_path = os.path.join(TEST_DIR, file)

        img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).flatten().reshape(1, -1)

        img_scaled = scaler.transform(img)
        img_pca    = pca.transform(img_scaled)

        pred = model.predict(img_pca)[0][0]
        label = class_names[int(pred > 0.5)]
        conf = pred if pred > 0.5 else 1-pred

        print(f"{file:<20} ---> {label} ({conf*100:.2f}%)")

print("\n====================================================\n")
