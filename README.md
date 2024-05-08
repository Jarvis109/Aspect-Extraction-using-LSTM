# Aspect-Extraction using LSTM

This repository contains code for performing Aspect-Based Sentiment Analysis (ABSA) using LSTM (Long Short-Term Memory) networks. ABSA is a subtask of sentiment analysis that focuses on identifying aspects or attributes of a product or service mentioned in a text and determining the sentiment associated with each aspect.

## Overview

The code in this repository is designed to perform ABSA on a dataset of laptop reviews. It utilizes LSTM networks to predict the sentiment of each aspect mentioned in the reviews.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- scikit-learn
- keras

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/aspect-based-sentiment-analysis.git
```

### Usage

1. Run `aspect_sentiment_analysis.ipynb` to preprocess the data, train the LSTM model, and evaluate its performance.

2. Modify the file paths and parameters as needed to suit your dataset or requirements.

## File Descriptions

- `aspect_sentiment_analysis.ipynb`: Jupyter notebook containing the code for preprocessing the data, training the LSTM model, and evaluating its performance.

- `train.json`: JSON file containing the training data. Each entry represents a review with associated aspects and sentiment.

## Data Preparation

The training data is loaded from the `train.json` file. Each entry in the JSON file contains tokens (words), aspects (attributes), and BIO tags (Begin, Inside, Outside) indicating the aspect boundaries.

## Model Architecture

The LSTM model architecture is as follows:

1. Embedding layer: Converts tokens into dense vectors of fixed size.
2. LSTM layer: Processes sequences of input data.
3. Dropout layer: Regularizes the model to prevent overfitting.
4. TimeDistributed layer: Applies a Dense layer to each time step of the input sequence.

## Training and Evaluation

The model is trained using the training data and evaluated using a validation set. The final performance is evaluated on a separate test set.

## Results

After training, the model's performance is evaluated on the test set. The final test loss and accuracy are printed.

