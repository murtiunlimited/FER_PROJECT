import tensorflow as tf
import numpy as np
import cv2
import os

# Load your trained model from root folder
MODEL_PATH = os.path.join("../final_emotion_model.keras")
CLASS_NAMES = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_face(face_img):
    face = cv2.resize(face_img, (48,48))
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)
    class_idx = np.argmax(preds)
    return CLASS_NAMES[class_idx]