import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (48, 48)
NUM_CLASSES = 7

def build_light_model():
    inputs = tf.keras.Input(shape=(48, 48, 1))

    # Block 1
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(7, activation='softmax')(x)
    return models.Model(inputs, outputs)