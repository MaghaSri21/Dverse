# Dverse


## DV1
# Parkinson's Disease Classification

## Overview
This Jupyter notebook explores machine learning techniques to classify Parkinson's disease based on biomedical voice measurements.

## Dataset
The dataset used is sourced from the UCI Machine Learning Repository:
- **Parkinson's Disease Dataset** ([Link](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data))

## Implemented Techniques
- **Data Preprocessing**
  - Standardization using `StandardScaler`
  - Train-test splitting
- **Machine Learning Models**
  - K-Nearest Neighbors (KNN)
  - Convolutional Neural Network (CNN) with `Keras`
- **Evaluation Metrics**
  - Accuracy
  - Precision, Recall, F1-score
  - Classification Report

## Dependencies
Install the required libraries using:
```bash
pip install numpy pandas scikit-learn keras tensorflow jupyter
```



## DV2
# Hand Landmark Detection using MediaPipe and OpenCV

## Overview
This Jupyter notebook demonstrates real-time hand landmark detection using **MediaPipe Hands** and **OpenCV**. The application captures video from a webcam, detects hand landmarks, and overlays them on the live video feed.

## Features
- Real-time **hand detection and tracking**.
- **Draws landmarks** on detected hands.
- **Displays landmark indexes** for each point on the hand.
- Uses **OpenCV** for video capture and rendering.

## Dependencies
To run this notebook, install the required dependencies using:
```bash
pip install opencv-python mediapipe jupyter
```



## DV3
# FAQ Chatbot using TF-IDF and BERT

## Overview
This Jupyter Notebook implements an FAQ chatbot using **two approaches**:
1. **TF-IDF & Cosine Similarity**: A lightweight method for finding relevant responses from an FAQ dataset.
2. **BERT-based Model**: A deep learning approach for improved question-answer matching.

## Features
- Loads and processes an **FAQ dataset**.
- **TF-IDF-based chatbot** that finds the most relevant answer using cosine similarity.
- **BERT-powered chatbot** for advanced question-answering.
- **Interactive chat interface** where users can ask questions.

## Dependencies
Install the required libraries using:
```bash
pip install nltk pandas scikit-learn transformers torch
```
