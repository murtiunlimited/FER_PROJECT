import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# =========================
# Paths & Config
# =========================
MODEL_PATH = os.path.join("final_emotion_model.keras")  # Trained model
IMG_SIZE = (48, 48)
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# Load Model & Face Detector
# =========================
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =========================
# Preprocessing
# =========================
def preprocess_face(face_img):
    """
    Convert face to grayscale, resize, normalize and expand dimensions for model input
    """
    face = cv2.resize(face_img, IMG_SIZE)
    face = face / 255.0                     # Normalize
    face = np.expand_dims(face, axis=-1)    # Add channel: (48,48,1)
    face = np.expand_dims(face, axis=0)     # Add batch: (1,48,48,1)
    return face

# =========================
# Prediction
# =========================
def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    return CLASS_NAMES[class_idx]

# =========================
# Webcam Loop
# =========================
def run_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            label = predict_emotion(face_gray)

            # Draw rectangle & label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Webcam Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    run_webcam()