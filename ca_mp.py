import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import math

# ==================== Configuration ====================
class Config:
    MODEL_PATH = 'models/attention_cnn.h5'
    IMG_SIZE = (128, 128)
    CLASSES = ['Attentive', 'Distracted']
    COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for Attentive, Red for Distracted
    
    # Face Mesh Config
    MAX_FACES = 10 # Reduced for performance on normal laptop
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Drowsiness Config
    EAR_THRESHOLD = 0.2
    MAR_THRESHOLD = 0.5
    
    # Visuals
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SMOOTHING_FRAMES = 5

# Landmarks indices for EAR/MAR/Gaze
RIGHT_EYE = [33, 133, 160, 144, 159, 145, 158, 153] 
LEFT_EYE = [263, 362, 387, 373, 386, 374, 385, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ... existing utilities ...

def calculate_gaze(landmarks, eye_indices, iris_indices, w, h):
    lms = landmarks.landmark
    p = lambda i: (lms[i].x * w, lms[i].y * h)
    
    eye_center_x = (p(eye_indices[0])[0] + p(eye_indices[1])[0]) / 2
    iris_center_x = sum([p(i)[0] for i in iris_indices]) / 4
    
    # Normalized horizontal offset of iris within eye
    # 0 = Center, Negative = Looking Left, Positive = Looking Right
    eye_width = distance(p(eye_indices[0]), p(eye_indices[1]))
    if eye_width == 0: return 0
    return (iris_center_x - eye_center_x) / eye_width

# ==================== Utilities ====================
class PredictionSmoother:
    def __init__(self, size=Config.SMOOTHING_FRAMES):
        self.history = {}
        self.size = size

    def smooth(self, face_id, prediction):
        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=self.size)
        self.history[face_id].append(prediction)
        return sum(self.history[face_id]) / len(self.history[face_id])

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(landmarks, eye_indices, w, h):
    lms = landmarks.landmark
    p = lambda i: (lms[i].x * w, lms[i].y * h)
    
    # Vertical distances
    v1 = distance(p(eye_indices[2]), p(eye_indices[3]))
    v2 = distance(p(eye_indices[4]), p(eye_indices[5]))
    v3 = distance(p(eye_indices[6]), p(eye_indices[7]))
    # Horizontal distance
    horiz = distance(p(eye_indices[0]), p(eye_indices[1]))
    
    ear = (v1 + v2 + v3) / (3.0 * horiz)
    return ear

def calculate_mar(landmarks, mouth_indices, w, h):
    lms = landmarks.landmark
    p = lambda i: (lms[i].x * w, lms[i].y * h)
    
    v1 = distance(p(mouth_indices[2]), p(mouth_indices[3]))
    v2 = distance(p(mouth_indices[4]), p(mouth_indices[5]))
    v3 = distance(p(mouth_indices[6]), p(mouth_indices[7]))
    horiz = distance(p(mouth_indices[0]), p(mouth_indices[1]))
    
    return (v1 + v2 + v3) / (3.0 * horiz)

def get_face_bbox(landmarks, frame_w, frame_h):
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    w, h = xmax - xmin, ymax - ymin
    
    # Margin
    xmin = max(0, xmin - w * 0.1)
    xmax = min(1, xmax + w * 0.1)
    ymin = max(0, ymin - h * 0.2)
    ymax = min(1, ymax + h * 0.1)
    
    return (int(xmin * frame_w), int(ymin * frame_h), 
            int((xmax - xmin) * frame_w), int((ymax - ymin) * frame_h))

# ==================== Main Application ====================
def main():
    print(f"Loading Model: {Config.MODEL_PATH}...")
    try:
        model = load_model(Config.MODEL_PATH)
        print("✓ Model loaded")
    except:
        print("✗ Model load failed. Ensure models/attention_cnn.h5 exists.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=Config.MAX_FACES,
        min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
        refine_landmarks=True
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    smoother = PredictionSmoother()
    cap = cv2.VideoCapture(0)

    print("\nStarting Classroom Monitor...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        counts = {'Attentive': 0, 'Distracted': 0, 'Drowsy': 0}

        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # 1. Landmarks Visual
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                # 2. Features (EAR/MAR/GAZE)
                ear = (calculate_ear(face_landmarks, RIGHT_EYE, w, h) + 
                       calculate_ear(face_landmarks, LEFT_EYE, w, h)) / 2
                mar = calculate_mar(face_landmarks, MOUTH, w, h)
                gaze_x = (calculate_gaze(face_landmarks, LEFT_EYE, LEFT_IRIS, w, h) + 
                          calculate_gaze(face_landmarks, RIGHT_EYE, RIGHT_IRIS, w, h)) / 2
                
                # Robust Head Pose (Ratio of distances between nose and eyes)
                # This is less sensitive to face position in the frame
                nose_x = face_landmarks.landmark[1].x
                l_eye_x = face_landmarks.landmark[33].x
                r_eye_x = face_landmarks.landmark[263].x
                face_width = abs(r_eye_x - l_eye_x)
                
                # head_turn > 0 means looking right, < 0 means looking left
                # Normalized by face width to be scale-invariant
                head_turn = (nose_x - (l_eye_x + r_eye_x) / 2) / face_width if face_width > 0 else 0

                # 3. CNN Classification (Secondary Hint)
                bx, by, bw, bh = get_face_bbox(face_landmarks, w, h)
                face_crop = frame[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
                
                status = "ATTENTIVE" # Default State
                color = (0, 255, 0)   # Default Color (Green)
                cnn_hint = ""
                
                if face_crop.size > 0:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, Config.IMG_SIZE)
                    face_input = np.expand_dims(face_resized / 255.0, axis=0)
                    raw_pred = model.predict(face_input, verbose=0)[0][0]
                    pred = smoother.smooth(idx, raw_pred)
                    cnn_hint = f" (CNN: {pred:.2f})"
                    
                    # Distraction Detection (LANDMARK BASED - RELIABLE)
                    if mar > 0.8: # Yawning Check (High Priority)
                        status = "YAWNING"
                        color = (255, 255, 0) # Cyan/Yellow-ish
                        counts['Distracted'] += 1
                    elif ear < 0.15: 
                        status = "DROWSY"
                        color = (0, 165, 255) # Orange
                        counts['Drowsy'] += 1
                    elif abs(head_turn) > 0.4: 
                        status = "NOT FOCUS (HEAD)"
                        color = (0, 0, 255) # Red
                        counts['Distracted'] += 1
                    elif abs(gaze_x) > 0.25: 
                        status = "LOOKING AWAY"
                        color = (255, 0, 255) # Purple
                        counts['Distracted'] += 1
                    else:
                        status = "ATTENTIVE"
                        color = (0, 255, 0)
                        counts['Attentive'] += 1

                    # Drawing
                    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), color, 2)
                    cv2.putText(frame, f"{status}{cnn_hint}", (bx, by-10), Config.FONT, 0.5, color, 2)
                    
                    # Visual debug HUD
                    cv2.putText(frame, f"E:{ear:.2f} G:{gaze_x:.2f} H:{head_turn:.2f} M:{mar:.2f}", 
                                (bx, by+bh+15), Config.FONT, 0.4, (255, 255, 255), 1)

        # UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        total = sum(counts.values())
        rate = (counts['Attentive'] / total * 100) if total > 0 else 0
        
        cv2.putText(frame, f"Total Students: {total}", (10, 30), Config.FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Attentive: {counts['Attentive']}", (10, 60), Config.FONT, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Attention Rate: {rate:.0f}%", (10, 90), Config.FONT, 0.7, (255, 255, 0), 2)

        cv2.imshow('Classroom Attention Monitor (MediaPipe)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
