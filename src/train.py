import os
import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model import build_light_model

BASE_DIR = "data/processed"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(x, y):
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = tf.image.per_image_standardization(x)
    return x, y

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels='inferred', label_mode='categorical', color_mode='grayscale',
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
).map(preprocess_image).map(lambda x, y: (data_augmentation(x), y)).prefetch(tf.data.AUTOTUNE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, labels='inferred', label_mode='categorical', color_mode='grayscale',
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
).map(preprocess_image).prefetch(tf.data.AUTOTUNE)

model = build_light_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_emotion_model.keras", monitor="val_accuracy", save_best_only=True)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
model.save("final_emotion_model.keras")