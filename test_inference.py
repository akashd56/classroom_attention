import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import os

# Configuration
MODEL_PATH = 'models/attention_cnn.h5'
IMG_SIZE = (128, 128)
CLASSES = ['Attentive', 'Distracted']

def test_on_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    print(f"\nTesting on: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not read image")
        return

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Detection
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_frame)

    if results.detections:
        print(f"Detected {len(results.detections)} faces.")
        model = load_model(MODEL_PATH)
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            x, y = max(0, x), max(0, y)
            
            face_img = frame[y:y+bh, x:x+bw]
            if face_img.size == 0: continue
            
            face_img_resized = cv2.resize(face_img, IMG_SIZE)
            face_img_normalized = face_img_resized / 255.0
            face_input = np.expand_dims(face_img_normalized, axis=0)

            prediction = model.predict(face_input, verbose=0)[0][0]
            class_idx = 1 if prediction > 0.5 else 0
            label = CLASSES[class_idx]
            print(f"Prediction: {label} ({prediction:.4f})")
    else:
        print("No faces detected.")

if __name__ == "__main__":
    test_on_image('dataset/attentive/NATURAL_1000.png')
    test_on_image('dataset/distracted/DROWSY_1000.png')
