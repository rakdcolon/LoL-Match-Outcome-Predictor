# League of Legends Match Outcome Predictor

This project builds a machine learning model to **predict the winner of a League of Legends match** based on pre-game and early-game data. It demonstrates the full ML pipeline — from data preprocessing and feature engineering to model training, evaluation, and interpretation.

---

## Project Overview

The goal of this project is to use player statistics to predict whether the **Blue team** or **Red team** will win a given match of League of Legends.
We use historical match data from public Kaggle datasets and train a classifier to learn key patterns that lead to victory.

---

## Features

* **Data Processing:** Cleaned and preprocessed raw match data (handle missing values, encode categorical features, normalize numeric stats).
* **Model Training:** Trained a decision tree machine learning model to predict match outcomes.
* **Evaluation:** Measured performance using accuracy, precision, recall, F1-score, and confusion matrix.
* **Feature Importance:** Understood which features most influence the outcome.

---

## Dataset

The model is trained on publicly available match data such as:

* [League of Legends Ranked Matches Dataset (Kaggle)](https://www.kaggle.com/)

---

## Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.99  |
| Precision | 0.99  |
| Recall    | 0.99  |
| F1-score  | 0.99  |

Sound too good to be true? I know.. but it is! No data leakage nor cheating occurs and the model extrapolates very well!

---
