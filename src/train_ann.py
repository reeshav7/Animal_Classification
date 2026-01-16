import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --------- Settings ---------
IMG_SIZE = 64
PCA_MAX_COMPONENTS = 600
EPOCHS = 40
BATCH_SIZE = 64

TRAIN_DIR = "../data/train"   
TEST_DIR  = "../data/test"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --------- Load images and labels (train) ---------
def load_dataset_from_folder(folder_path, img_size):
    X, y, class_names = [], [], []
    # detect class folders (sorted to keep label order consistent)
    classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    for idx, cls in enumerate(classes):
        cls_folder = os.path.join(folder_path, cls)
        for fname in os.listdir(cls_folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".webp")):
                continue
            try:
                img = Image.open(os.path.join(cls_folder, fname)).convert("L").resize((img_size, img_size))
                X.append(np.array(img).flatten())
                y.append(idx)
            except Exception as e:
                # skip unreadable images
                print(f"Skipped {fname} in {cls}: {e}")
        class_names.append(cls)
    X = np.array(X)
    y = np.array(y)
    return X, y, classes

print("\nLoading TRAIN dataset...")
X_train, y_train, class_names = load_dataset_from_folder(TRAIN_DIR, IMG_SIZE)
print(f"Train samples: {X_train.shape[0]}  |  Classes: {len(class_names)} -> {class_names}")

print("\nLoading TEST dataset...")
X_test, y_test, _ = load_dataset_from_folder(TEST_DIR, IMG_SIZE)
print(f"Test samples: {X_test.shape[0]}")

# Basic checks
num_classes = len(class_names)
if num_classes < 2:
    raise SystemExit("Need at least 2 classes to train.")

if X_train.shape[0] < num_classes:
    print("Warning: number of training samples is small relative to classes.")

# --------- Scale & PCA (fit on train only) ---------
print("\nScaling data...")
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# determine reasonable n_components for PCA
n_samples, n_features = X_train_s.shape
n_components = min(PCA_MAX_COMPONENTS, n_samples - 1, n_features)
if n_components <= 0:
    raise SystemExit("Not enough samples for PCA. Increase dataset size.")

print(f"Applying PCA with n_components = {n_components} (samples={n_samples}, features={n_features})")
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)

# save scaler + pca for later inference
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_multiclass.pkl"))
joblib.dump(pca, os.path.join(MODEL_DIR, "pca_multiclass.pkl"))

# --------- Prepare labels for multi-class ----------
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat  = to_categorical(y_test, num_classes=num_classes)

# --------- Build ANN model ----------
print("\nBuilding ANN model...")
model = Sequential([
    Input(shape=(n_components,)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0008),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --------- Train ----------
print("\nTraining ANN (multi-class)...")
history = model.fit(
    X_train_pca, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_pca, y_test_cat),
    verbose=1
)

# save model
model_path = os.path.join(MODEL_DIR, "ann_model_multiclass.h5")
model.save(model_path)
print(f"\nModel saved to {model_path}")

# --------- Evaluation ----------
print("\nEvaluating on test set...")
y_pred_prob = model.predict(X_test_pca)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# plot accuracy & loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend(); plt.title("Loss")
plt.tight_layout()
plt.show()

# save final model artifacts list
print("\nSaved artifacts:")
print(f" - Model: {model_path}")
print(f" - Scaler: {os.path.join(MODEL_DIR,'scaler_multiclass.pkl')}")
print(f" - PCA: {os.path.join(MODEL_DIR,'pca_multiclass.pkl')}")

