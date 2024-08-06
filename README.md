# Amazon Alexa Reviews Sentiment Analysis

This repository contains an end-to-end Natural Language Processing (NLP) project for sentiment analysis on Amazon Alexa reviews. The project involves data exploration, preprocessing, model building, and deploying a Flask web application for sentiment prediction.

## Project Overview

The goal of this project is to build a sentiment analysis classifier to predict whether a given review is positive, neutral, or negative. The project follows an end-to-end approach, starting from data exploration to deploying a machine learning model as a Flask API.

## Dataset

The dataset used in this project consists of Amazon Alexa reviews. It is publicly available on Kaggle and other platforms. The dataset includes the following columns:

- `rating`: The rating given by the user (1 to 5 stars)
- `date`: The date of the review
- `variation`: The variation of the product
- `verified_reviews`: The review text
- `feedback`: Whether the review is positive (1) or negative (0)

## Data Exploration

Initial data exploration involves checking for null values, understanding the distribution of ratings, and analyzing the length of the reviews. Key insights include:

- Most reviews are positive, with a rating of 5 stars.
- Positive reviews tend to be longer and more detailed.
- A word cloud is generated to visualize the most frequent words in positive and negative reviews.

## Data Preprocessing

Data preprocessing steps include:

- Cleaning the text data by removing non-alphabet characters.
- Converting text to lowercase and tokenizing.
- Removing stop words and applying stemming using Porter Stemmer.
- Vectorizing the text data using CountVectorizer.

## Model Building

Several classification models are built and evaluated:

- **Random Forest Classifier**
- **XGBoost Classifier**
- **Decision Tree Classifier**

Hyperparameter tuning is performed using GridSearchCV to find the best model. The XGBoost model is selected for deployment due to its superior performance.

## Model Evaluation

The models are evaluated based on their training and test accuracy. The final selected model (XGBoost) achieves:

- Training Accuracy: 97%
- Test Accuracy: 94%

## Flask Application

The final model is deployed as a Flask web application. The application provides an API endpoint to predict the sentiment of a given review. The Flask app structure includes:

- `app.py`: The main Flask application file.
- `model.pkl`: The saved machine learning model.
- `vectorizer.pkl`: The saved CountVectorizer model.

## How to Run

1. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Flask application:
    ```bash
    python app.py
    ```

4. Open your web browser and navigate to `http://127.0.0.1:5000` to access the sentiment analysis API.

## Dependencies

The project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- wordcloud
- xgboost
- Flask
- pickle

