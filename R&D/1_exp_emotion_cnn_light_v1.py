import warnings
warnings.filterwarnings("ignore")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report


# =========================
# PATHS & CONFIG
# =========================
BASE_DIR = "preprocessed_images"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")


IMG_SIZE = (48, 48)
BATCH_SIZE = 32  # smaller batch reduces RAM
EPOCHS = 50


CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# =========================
# DATA LOADING
# =========================
def preprocess_image(x, y):
   # Convert to float32
   x = tf.image.convert_image_dtype(x, tf.float32)
   # Standardize to zero mean, unit variance
   x = tf.image.per_image_standardization(x)
   return x, y


# Light augmentation
data_augmentation = tf.keras.Sequential([
   layers.RandomFlip("horizontal"),
   layers.RandomRotation(0.1),
   layers.RandomZoom(0.1)
])


train_ds = tf.keras.utils.image_dataset_from_directory(
   TRAIN_DIR,
   labels='inferred',
   label_mode='categorical',
   color_mode='grayscale',
   image_size=IMG_SIZE,
   batch_size=BATCH_SIZE,
   shuffle=True
).map(preprocess_image).map(lambda x, y: (data_augmentation(x), y)).prefetch(tf.data.AUTOTUNE)


val_ds = tf.keras.utils.image_dataset_from_directory(
   VAL_DIR,
   labels='inferred',
   label_mode='categorical',
   color_mode='grayscale',
   image_size=IMG_SIZE,
   batch_size=BATCH_SIZE,
   shuffle=False
).map(preprocess_image).prefetch(tf.data.AUTOTUNE)


# =========================
# MODEL
# =========================
def build_light_model():
   inputs = tf.keras.Input(shape=(48, 48, 1))


   # Block 1
   x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
   x = layers.MaxPooling2D((2, 2))(x)


   # Block 2
   x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
   x = layers.MaxPooling2D((2, 2))(x)


   # Fully connected
   x = layers.Flatten()(x)
   x = layers.Dense(128, activation='relu')(x)
   x = layers.Dropout(0.3)(x)


   outputs = layers.Dense(7, activation='softmax')(x)
   return models.Model(inputs, outputs)


model = build_light_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# =========================
# CALLBACKS
# =========================
callbacks = [
   EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
   ModelCheckpoint("best_emotion_model_light_preprocessed.keras", monitor="val_accuracy", save_best_only=True)
]


# =========================
# TRAINING
# =========================
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)


# =========================
# EVALUATION
# =========================
y_true, y_pred = [], []


for images, labels in val_ds:
   preds = model.predict(images, verbose=0)
   y_true.extend(tf.argmax(labels, axis=1).numpy())
   y_pred.extend(tf.argmax(preds, axis=1).numpy())


print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


# =========================
# SAVE MODEL
# =========================
model.save("final_emotion_model_light_preprocessed.keras")
