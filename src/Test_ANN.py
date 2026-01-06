import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import joblib

# ============================
# Paths
# ============================
MODEL_PATH = "models/ann_model_multiclass.h5"
SCALER_PATH = "models/scaler_multiclass.pkl"
PCA_PATH    = "models/pca_multiclass.pkl"
TEST_DIR    = "../test_images"

IMG_SIZE = 64   # must match training

# ============================
# Load model, scaler, PCA
# ============================
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca    = joblib.load(PCA_PATH)

# get class names from train folder (consistent indexing)
train_dir = "../data/train"
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

print("\n========== ANN MULTI-CLASS BATCH TEST ==========\n")

# ============================
# Run batch test
# ============================
for file in os.listdir(TEST_DIR):
    if file.lower().endswith((".jpg",".jpeg",".png",".jfif",".bmp",".webp")):
        img_path = os.path.join(TEST_DIR, file)

        img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).flatten().reshape(1, -1)

        # scaling + PCA
        img_s = scaler.transform(img)
        img_p = pca.transform(img_s)

        # prediction
        pred = model.predict(img_p, verbose=0)[0]
        idx  = np.argmax(pred)
        conf = pred[idx] * 100

        print(f"{file:<30} --> {class_names[idx].upper():<12} ({conf:.2f}%)")

print("\n===============================================\n")
