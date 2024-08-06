This project focuses on detecting handwritten text from unstructured data and accurately predicting the detected text using a deep learning model. The detection model uses two datasets: one of handwritten text and the other of digital text. The prediction model combines a custom Convolutional Neural Network (CNN) with two Long Short-Term Memory (LSTM) layers. The final model is integrated with Streamlit to provide an interactive user interface.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)

## Project Overview
The goal of this project is to develop a deep learning model capable of detecting and predicting handwritten text from unstructured data. The project includes the following steps:
1. Data collection and preprocessing from handwritten and digital text datasets.
2. Building a custom CNN model for text detection.
3. Building a custom CNN+2LSTM model for text prediction.
4. Training and evaluating the models.
5. Integrating the models with a Streamlit interface for real-time predictions.

## Model Architecture
The model architecture includes:
- A custom Convolutional Neural Network (CNN) for detecting text from both handwritten and digital text datasets.
- Another custom CNN combined with two Long Short-Term Memory (LSTM) layers for predicting the detected text.
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
