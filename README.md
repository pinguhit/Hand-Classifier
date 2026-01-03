# Hand Gesture Classifier using MobileNetV2

## Overview
This project implements a real-time hand gesture classification system capable of recognizing three hand states: open, closed, and no gesture. The system uses transfer learning with MobileNetV2 and an end-to-end computer vision pipeline built using Python and OpenCV.

To improve real-world performance, gesture hysteresis and temporal smoothing were implemented to reduce prediction noise and ensure stable gesture recognition during real-time inference.

## Features
- Real-time hand gesture classification
- Transfer learning using MobileNetV2
- Custom dataset creation and preprocessing
- Gesture hysteresis to prevent rapid class switching
- Temporal smoothing window for stable predictions
- End-to-end pipeline from data collection to evaluation

## Model and Approach
- Model Architecture: MobileNetV2 (pretrained and fine-tuned)
- Classes:
  - Open Hand
  - Closed Hand
  - No Gesture
- Training Strategy:
  - Feature extraction using pretrained layers
  - Layer freezing and selective unfreezing for fine-tuning
  - Supervised learning with labeled image data

## Pipeline
1. Dataset collection and labeling
2. Image preprocessing and resizing
3. Model training using transfer learning
4. Model evaluation and testing
5. Real-time inference using webcam input


## Use Cases
- Human-computer interaction
- Gesture-based control systems
- Assistive technologies
- Real-time computer vision applications

## Future Improvements
- Expand gesture vocabulary
- Integrate hand landmark-based hybrid models
- Optimize inference latency
- Deploy as a desktop or web application


