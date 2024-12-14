This project focuses on Hand-Written Text Recognition and Identification, aimed at converting
unstructured handwritten documents like bank cheques and transcripts into structured digital data.
Using a combination of CNN and BiLSTM, the system identifies and extracts handwritten text
effectively. Data for training includes handwritten words from the IAM dataset and bank cheques
from various sources. Preprocessing of cheques involves making horizontal lines in cheques
transparent for precise contour detection and to avoid large contours. The CNN model for
differentiation between digital text and hand-written text, processes contours and achieves over 90%
accuracy, while the BiLSTM-based identification model achieves 70% accuracy with good
predictions. Despite mixed results, the prototype demonstrates significant potential for automating 
text extraction. Further research is required to enhance performance and expand practical
applications. This work highlights the importance of deep learning in document digitization.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)

## Project Overview
The goal of this project is to develop a deep learning model capable of detecting and predicting handwritten text from unstructured data. The project includes the following steps:
1. Data collection and preprocessing from handwritten and digital text datasets.
2. Building a custom CNN model for text identification(HandWritten or Digital Text).
3. Building a custom CNN+2BiLSTM model for text recognition(Accurately identifying the hand-written text) .
4. Training and evaluating the models.
5. Processing the documents using Computer Vision.
6. Integrating the models with a Streamlit interface for real-time predictions.

## Model Architecture
The model architecture includes:
- A custom Convolutional Neural Network (CNN) for detecting text from both handwritten and digital text datasets.
- Another custom CNN combined with two Birectional Long Short-Term Memory (BiLSTM) layers for predicting the detected text.
- Fully connected layers for final prediction.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https:/Rahulyzk-007/github.com//handwritten-text-detection.git
   cd handwritten-text-detection

## Usage
The Model can be used to retrieve the hand written text first from unstructured data and later accurately predict the Hand written text.
This comes into picture when processing checks transactions and other documents where pytesseract fails to predict hand written text.
By this model we can easily predict the entire text from a document (both Hand-Written and Digital)

## Datasets
Download the IAM words dataset (zipped) from the 'https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database'
