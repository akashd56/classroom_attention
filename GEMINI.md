# Classroom Attention Monitor - Implementation Details

This project is a streamlined classroom attention monitor designed for a university seminar. It uses MediaPipe for robust face detection and a TensorFlow Sequential CNN for behavior classification.

## 🚀 Quick Start (Using Virtual Environment)

The project includes a `Makefile` to handle all setup and execution steps within a Python virtual environment (`venv`).

1. **Setup Environment**:
   ```bash
   make setup
   ```
   *Creates a venv and installs all dependencies (TensorFlow, MediaPipe, OpenCV, etc.).*

2. **Run Real-Time Monitor**:
   ```bash
   make run
   ```
   *Starts the advanced FaceMesh-based monitor with drowsiness and yawning detection.*

## 🛠 Project Components

### 1. Real-Time Inference (`ca_mp.py`)
- **Detector**: MediaPipe FaceMesh (tracks up to 10 faces simultaneously).
- **Behavior Classification**: 
  - **Drowsiness**: Calculated via **EAR (Eye Aspect Ratio)**.
  - **Yawning**: Calculated via **MAR (Mouth Aspect Ratio)**.
  - **Attention Score**: CNN-based prediction (Attentive vs. Distracted) shown as a hint.
  - **Head Pose**: Monitors head turn to detect looking away.
- **Smoothing**: Prediction history per face to prevent flickering.

### 2. Folder Structure
```text
classroom_attention/
├── ca_mp.py             # Main application (FaceMesh + CNN)
├── Makefile             # Automation script
├── requirements.txt     # Dependencies
├── models/              # Saved model (.h5)
└── dataset/             # Image data (attentive/distracted)
```

## 🎓 Seminar Extensions
- **Visuals**: FaceMesh contours provide a high-tech "AI" look suitable for demos.
- **Metrics**: Real-time student counts in a simplified HUD.
- **Robustness**: Long-range face detection optimized for classroom seating.
