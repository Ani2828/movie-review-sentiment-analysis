# Movie Review Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project that classifies movie reviews as **Positive** or **Negative** using TF-IDF and Logistic Regression.

## Live Demo

🚀 **Try the application:** [Movie Review Sentiment Analysis](https://movie-review-sentiment-analysis-m24y9rxy2vx8gonnpbmthl.streamlit.app/)

Enter a movie review and get:
- Positive or Negative sentiment
- Prediction confidence
- Real-time prediction from the trained NLP model

## Results

| Metric | Score |
|---|---:|
| Test Accuracy | **89.52%** |
| Negative F1-score | **0.89** |
| Positive F1-score | **0.90** |

## Project Overview

The goal of this project is to build a machine learning model capable of understanding the sentiment expressed in movie reviews.

The project follows a complete NLP workflow:

1. Load the IMDb movie review dataset
2. Explore the dataset
3. Clean and preprocess text
4. Convert text into numerical features using TF-IDF
5. Train a Logistic Regression classifier
6. Evaluate the model
7. Test the model on custom reviews
8. Generate prediction confidence scores
9. Save the trained model and vectorizer

## Dataset

The project uses the IMDb Movie Reviews dataset.

- Training samples: 25,000
- Testing samples: 25,000
- Sentiment classes:
  - `0` → Negative
  - `1` → Positive

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Hugging Face Datasets

## NLP Pipeline

```text
Movie Review
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Sentiment Prediction
     ↓
Confidence Score
