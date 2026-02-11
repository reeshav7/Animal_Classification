import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

#--- Settings -------
IMG_SIZE = 64
PCA_COMPONENTS = 600
EPOCHS = 40
BATCH_SIZE = 64
DATA_DIR = "../data"

print("\nScanning dataset for images")
all_images, labels = [], []

for cl in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, cl)
    if os.path.isdir(path):
        for img_name in os.listdir(path):
            try:
                img = Image.open(os.path.join(path, img_name)).convert("L").resize((IMG_SIZE, IMG_SIZE))
                all_images.append(np.array(img).flatten())
                labels.append(cl)
            except:
                pass

all_images = np.array(all_images)
labels = np.array(labels)

print(f"Total Images Found: {len(all_images)}")
classes = np.unique(labels)
print("Detected Classes:", classes)

y = np.array([0 if l == classes[0] else 1 for l in labels])

# Split
X_train, X_test, y_train, y_test = train_test_split(all_images, y, test_size=0.2, stratify=y)

print("\nScaling & PCA reducing")
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)

print(f"PCA Output → {X_train_pca.shape}")

# --- ANN model ---------
model = Sequential([
    Input(shape=(PCA_COMPONENTS,)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0008),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nTraining ANN.")
history = model.fit(
    X_train_pca, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_pca, y_test)
)

model.save("models/ann_model.h5")
print("\nModel saved as ann_model.h5")

# ================================
# Evaluation
# ================================
pred = (model.predict(X_test_pca) > 0.5).astype(int).flatten()

print("\nClassification Report:\n")
print(classification_report(y_test, pred, target_names=list(classes)))

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.show()

# Accuracy & Loss curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend(); plt.title("Loss")
plt.show()

# -- For Testing ------

import joblib
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
