# Classroom Attention Monitor - Implementation Details

This project is a small-scale, visually impressive classroom attention monitor designed for a university seminar. It uses MediaPipe for robust face detection and a TensorFlow Sequential CNN for behavior classification.

## 🚀 Quick Start (Using Virtual Environment)

The project includes a `Makefile` to handle all setup and execution steps within a Python virtual environment (`venv`).

1. **Setup Environment**:
   ```bash
   make setup
   ```
   *Creates a venv and installs all dependencies (TensorFlow, MediaPipe, OpenCV, etc.).*

2. **Download & Organize Data**:
   ```bash
   make download
   ```
   *Uses `kagglehub` to download the Drowsiness Detection dataset and organizes it into `dataset/attentive` and `dataset/distracted`.*

3. **Train the Model**:
   ```bash
   make train
   ```
   *Trains a Sequential CNN on the collected face images.*

4. **Run Real-Time Monitor**:
   - **Basic Version** (Face Detection): `make run`
   - **Advanced Version** (FaceMesh + Drowsiness + Yawning): `make run-mp`

## 🛠 Project Components

### 1. Data Collection (`kagglehub` & `collect_data.py`)
- We use the `yasharjebraeily/drowsy-detection-dataset` from Kaggle via `kagglehub`.
- `download_datasets.py` automatically maps 'NATURAL' faces to 'Attentive' and 'DROWSY' to 'Distracted'.
- `collect_data.py` allows for manual data collection using **MediaPipe Face Detection** for high-quality crops.

### 2. Model Architecture
- **Type**: Sequential CNN (TensorFlow).
- **Input**: 128x128x3 (RGB face crops).
- **Layers**: 3x Conv2D/MaxPooling blocks followed by a Dense hidden layer and a Sigmoid output.

### 3. Real-Time Inference (`ca_mp.py`)
- **Detector**: MediaPipe FaceMesh (tracks up to 10-20 faces simultaneously).
- **Features**:
  - **Attention Score**: CNN-based prediction (Attentive vs. Distracted).
  - **Drowsiness**: Calculated via **EAR (Eye Aspect Ratio)**.
  - **Yawning**: Calculated via **MAR (Mouth Aspect Ratio)**.
  - **Smoothing**: Prediction history per face to prevent flickering.

### 4. Folder Structure
```text
classroom_attention/
├── ca_mp.py             # Main application (FaceMesh + CNN)
├── train_simple.py      # Training script
├── download_datasets.py # Kaggle downloader
├── collect_data.py      # Manual data collection tool
├── Makefile             # Automation script
├── requirements.txt     # Dependencies
├── models/              # Saved model (.h5)
└── dataset/             # Organized image data
    ├── attentive/
    └── distracted/
```

## 🎓 Seminar Extensions
- **Visuals**: FaceMesh contours provide a high-tech "AI" look suitable for demos.
- **Metrics**: Real-time class-wide attention percentage and student counts.
- **Robustness**: Long-range face detection (`model_selection=1`) optimized for classroom seating.
