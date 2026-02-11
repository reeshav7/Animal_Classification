import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ============================
# Dataset paths
# ============================
data_dir = "../data"
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "test")    # using your test as validation

img_size = (224,224)
batch_size = 16
epochs = 10    # increase to 20+ later

# ============================
# Data Loader
# ============================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.3,
    horizontal_flip=True,
).flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode="categorical", shuffle=True
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size,
    class_mode="categorical", shuffle=True
)

# ============================
# CNN Model (TRANSFER LEARNING)
# ============================
base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

print("\n========== Training CNN ==========\n")
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

model.save("../models/cnn_animals_mobilenet.h5")
print("\nModel saved as cnn_animals_mobilenet.h5")
