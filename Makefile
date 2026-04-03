# Variables
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.DEFAULT_GOAL := help
.PHONY: setup run run-mp help

help:
	@echo "Classroom Attention Monitor - Makefile"
	@echo "---------------------------------------"
	@echo "make setup               - Create venv and install dependencies"
	@echo "make run                 - Start real-time monitor (MediaPipe FaceMesh + Drowsiness)"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) ca_mp.py

run-mp: run


