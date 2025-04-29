# Assignment 3: Subreddit Classification

This project addresses the task of classifying Reddit comments into one of five political subreddits using various text classification models. The assignment consists of three separate tracks, each focusing on a different modeling approach.

## Task Overview

The goal is to train and evaluate models that can predict which subreddit a given Reddit comment belongs to. This is a five-class classification problem, with balanced data across the following classes:

- The_Donald
- Conservative
- politics
- Libertarian
- ChapoTrapHouse

## Dataset

The dataset includes three files:

- `train.csv` – Contains Reddit comments and their labels (used for training).
- `dev.csv` – Contains Reddit comments and their labels (used for validation).
- `test.csv` – Contains only Reddit comments (used for final evaluation).

Each entry includes:
- `id`: Unique identifier for the comment
- `text`: The Reddit comment text
- `label`: Subreddit label (only in train and dev sets)

## Tracks

### Track 1 – Classic Machine Learning

This track uses a combination of feature engineering and traditional machine learning classifiers:

- **TF-IDF + Elastic Net Logistic Regression**: Captures character- and word-level patterns.
- **Word2Vec + XGBoost**: Word embeddings are averaged and used with a gradient boosting model.

**Output file**: `track_1_test.csv`

### Track 2 – BERT-style Models

This track fine-tunes a transformer-based language model:

- **TF-IDF + RoBERTa**: RoBERTa-base was fine-tuned on the training data to classify the Reddit comments directly. Preprocessing included lowercasing, tokenization, and truncation/padding of input text.

**Output file**: `track_2_test.csv`

### Track 3 – Open Track

This track combines predictions from models in Track 1 and Track 2:

- **Stacked Ensemble**: Combines TF-IDF + Elastic Net and RoBERTa outputs using a meta-classifier. Probabilistic outputs from each base model are fed into a logistic regression to generate final predictions.

**Output file**: `track_3_test.csv`

## Evaluation

Models will be evaluated based on **Macro F1 Score** on the test set. Each track contributes equally to the final grade. If Track 3 is submitted, the top two performing tracks will be selected for final evaluation.

Baseline performance scores are provided in the assignment for reference, including a TF-IDF Logistic Regression model and a Random Classifier.

