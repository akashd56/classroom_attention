---
marp: true
theme: default
paginate: true
backgroundColor: #f0f4f8
---

# 🎓 Classroom Attention Monitor

### Deep Dive into Implementation & Architecture

**Tech Stack:** MediaPipe + TensorFlow + OpenCV

---

## 🏗️ System Architecture

1. **Geometric Layer (Landmarks)**
   - High-speed detection of physical behaviors.
   - EAR (Blinking), MAR (Yawning), Gaze.

2. **Semantic Layer (CNN)**
   - Analyzes facial expressions & subtle context.

3. **Logic Controller**
   - Resolves conflicts between layers.
   - **Landmarks take precedence** to reduce false positives.

---

## 🛠️ Data & Inference Pipeline

![alt text](Camera Feed Landmark-2026-04-03-044914.svg)

---

##### 🧠 Hybrid Decision Flow

## ![alt text](<Camera Feed Landmark-2026-04-03-045044.svg>)

---

## 📂 Component Interaction Map

![alt text](<Camera Feed Landmark-2026-04-03-045448.svg>)

---

## 📸 Phase 1: Data Collection

- **`download_datasets.py`**:
  - Downloads Drowsiness Dataset from Kaggle.
  - Maps `NATURAL` -> `Attentive`.
  - Maps `DROWSY` -> `Distracted`.

- **`collect_data.py`**:
  - Uses MediaPipe for real-time face harvesting.
  - Crops & resizes faces to $128 \times 128$.

---

## 🧠 Phase 2: CNN Training

- **Sequential CNN Architecture**:
  - 3x Conv2D Layers (Feature extraction).
  - 3x MaxPooling (Spatial reduction).
  - Dense/Dropout (Prevention of overfitting).

- **Data Augmentation**:
  - Rotations, shifts, and horizontal flips.
  - Increases model robustness to camera angles.

---

## 🧪 Phase 3: Testing & Validation

- **Training Metrics**:
  - Loss: `binary_crossentropy`.
  - Optimizer: `Adam`.
  - Validation Split: 20%.

- **Temporal Smoothing**:
  - Averages predictions over 5 frames.
  - Prevents "flickering" of the focus status.

---

## 💻 Code: Configuration Settings

The `Config` class centralizes all thresholds:

```python
class Config:
    MODEL_PATH = 'models/attention_cnn.h5'
    IMG_SIZE = (128, 128)
    EAR_THRESHOLD = 0.15   # Eyes closed
    MAR_THRESHOLD = 0.80   # Yawning
    MAX_FACES = 10         # Multi-student
    SMOOTHING_FRAMES = 5   # UI stability
```

---

## 👁️ Geometric Math (Syntax)

Uses **Lambda functions** for coordinate mapping:

```python
p = lambda i: (lms[i].x * w, lms[i].y * h)
```

**Eye Aspect Ratio (EAR)**:
Calculates vertical opening normalized by width.

```python
v1 = distance(p(eye_indices[2]), p(eye_indices[3]))
horiz = distance(p(eye_indices[0]), p(eye_indices[1]))
ear = (v1 + v2 + v3) / (3.0 * horiz)
```

---

## 🎯 Gaze Tracking Logic

**Normalized Horizontal Offset**:
Detects iris position relative to eye width.

```python
eye_center_x = (p(eye_indices[0])[0] + p(eye_indices[1])[0]) / 2
iris_center_x = sum([p(i)[0] for i in iris_indices]) / 4
eye_width = distance(p(eye_indices[0]), p(eye_indices[1]))
gaze_offset = (iris_center_x - eye_center_x) / eye_width
```

Trigger: `abs(gaze_offset) > 0.25`

---

## 🧠 CNN Inference Pipeline

Critical preprocessing steps in Python:

```python
# Fix: OpenCV (BGR) -> Model (RGB)
face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

# Resize & Normalize
face_resized = cv2.resize(face_rgb, (128, 128))
face_input = np.expand_dims(face_resized / 255.0, axis=0)

# Prediction
raw_pred = model.predict(face_input, verbose=0)[0][0]
```

---

## ⚖️ Decision Hierarchy (Status)

Priority switch ensures reliable detection:

1. **YAWNING** (MAR > 0.8) ➔ Cyan
2. **DROWSY** (EAR < 0.15) ➔ Orange
3. **NOT FOCUS** (Head > 0.4) ➔ Red
4. **LOOKING AWAY** (Gaze > 0.25) ➔ Purple
5. **ATTENTIVE** (Default) ➔ Green
