# Variables
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.PHONY: setup download train run run-mp collect-attentive collect-distracted help

help:
	@echo "Classroom Attention Monitor - Makefile"
	@echo "---------------------------------------"
	@echo "make setup               - Create venv and install dependencies"
	@echo "make download            - Download and organize Kaggle dataset"
	@echo "make train               - Train the Sequential CNN model"
	@echo "make run                 - Start basic webcam inference (MediaPipe FaceDetection)"
	@echo "make run-mp              - Start advanced webcam inference (MediaPipe FaceMesh + Drowsiness)"
	@echo "make collect-attentive   - Manually collect 'attentive' face data"
	@echo "make collect-distracted  - Manually collect 'distracted' face data"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download:
	$(PYTHON) download_datasets.py

train:
	$(PYTHON) train_simple.py

run:
	$(PYTHON) ca_mp.py

run-mp:
	$(PYTHON) ca_mp.py

