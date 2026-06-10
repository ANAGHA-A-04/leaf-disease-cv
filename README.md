# Plant Leaf Disease Detector

## PyTorch Environment Setup

This repository contains the initial PyTorch workspace setup for the Plant Leaf Disease Detector project.

## Device Verification

PyTorch Version: 2.12.0+cpu

CUDA Available: False

Device Name: CPU

## Project Structure
leaf-disease-cv/
│
├── data/
│   ├── processed/
│   ├── raw/
│   ├── train/
│   │   ├── early_blight/
│   │   ├── healthy/
│   │   ├── late_blight/
│   │   └── leaf_mold/
│   └── val/
│       ├── early_blight/
│       ├── healthy/
│       ├── late_blight/
│       └── leaf_mold/
│
├── models/
├── notebooks/
│
├── src/
│   ├── convolution.py
│   ├── dataset.py
│   ├── devicecheck.py
│   └── visualize_batch.py
│
├── venv/
├── .gitignore
├── README.md
├── requirements.txt
├── sample_batch.png
└── split.py

## Dependencies

* torch
* torchvision
* Pillow
* matplotlib

## Class Imbalance Analysis
  Late Blight is the largest class with 1527 samples .

  Leaf Mold is the smallest class with 761 samples .

  Healthy contains 1273 samples.

  Early Blight contains 800 samples .

Observation: The dataset contains more samples for Late Blight and Healthy classes than for Early Blight and Leaf Mold, indicating a moderate level of class imbalance