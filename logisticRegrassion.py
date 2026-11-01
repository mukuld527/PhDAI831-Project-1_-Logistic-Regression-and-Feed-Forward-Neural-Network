"""
Project 1: Logistic Regression for Sentiment Analysis

Features:
1. Count of positive words
2. Count of negative words
3. Presence of "no" (1/0)
4. Count of first/second person pronouns
5. Presence of "!" (1/0)
6. log(word count)

Target:
1 = Positive Sentiment
0 = Negative Sentiment
"""

import logging
import numpy as np

# -----------------------------------------------------
# Logging Configuration
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------------------------------
# Training Dataset
# -----------------------------------------------------
# Example sentiment data:
#
# [positive_words,
#  negative_words,
#  no_present,
#  pronoun_count,
#  exclamation_present,
#  log_word_count]
#

X = np.array([
    [3, 0, 0, 1, 1, 2.30],  # Positive
    [2, 0, 0, 2, 1, 2.10],  # Positive
    [0, 3, 0, 1, 0, 2.00],  # Negative
    [0, 4, 1, 0, 0, 2.20],  # Negative
    [2, 1, 0, 1, 1, 2.00],  # Positive
    [1, 2, 1, 1, 0, 1.90],  # Negative
    [4, 0, 0, 2, 1, 2.50],  # Positive
    [0, 5, 1, 0, 0, 2.30],  # Negative
])

y = np.array([
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0
])

# -----------------------------------------------------
# Sigmoid Activation Function
# -----------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------------------------------
# Binary Cross Entropy Loss
# -----------------------------------------------------

def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


# -----------------------------------------------------
# Gradient Descent Training
# -----------------------------------------------------

def train_logistic_regression(
    X,
    y,
    learning_rate=0.01,
    epochs=5000
):

    m, n = X.shape

    # Initialize weights and bias
    weights = np.zeros(n)
    bias = 0.0

    logging.info("Training started")
    logging.info("Samples=%s Features=%s", m, n)

    for epoch in range(epochs):

        # Forward Propagation
        z = np.dot(X, weights) + bias
        y_hat = sigmoid(z)

        # Compute Gradients
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)

        # Update Parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Logging every 500 epochs
        if epoch % 500 == 0:

            loss = compute_loss(y, y_hat)

            logging.info(
                "Epoch=%s Loss=%.6f",
                epoch,
                loss
            )

    logging.info("Training completed")

    return weights, bias


# -----------------------------------------------------
# Train Model
# -----------------------------------------------------

weights, bias = train_logistic_regression(
    X,
    y,
    learning_rate=0.05,
    epochs=5000
)

# -----------------------------------------------------
# Display Learned Weights
# -----------------------------------------------------

feature_names = [
    "Positive Word Count",
    "Negative Word Count",
    "No Present",
    "Pronoun Count",
    "Exclamation Present",
    "Log Word Count"
]

logging.info("Learned Model Parameters")

for feature, weight in zip(feature_names, weights):
    logging.info("%s = %.4f", feature, weight)

logging.info("Bias = %.4f", bias)

# -----------------------------------------------------
# Example 1:
# Classification of New Positive Review
# -----------------------------------------------------

new_review_positive = np.array([
    2,      # positive words
    0,      # negative words
    0,      # no
    1,      # pronouns
    1,      # !
    1.79    # log(word count)
])

positive_probability = sigmoid(
    np.dot(new_review_positive, weights) + bias
)

logging.info(
    "Positive Review Probability = %.4f",
    positive_probability
)

if positive_probability >= 0.5:
    logging.info(
        "Prediction = Positive Sentiment"
    )
else:
    logging.info(
        "Prediction = Negative Sentiment"
    )

# -----------------------------------------------------
# Example 2:
# Classification of New Negative Review
# -----------------------------------------------------

new_review_negative = np.array([
    0,      # positive words
    2,      # negative words
    0,      # no
    0,      # pronouns
    0,      # !
    1.39    # log(word count)
])

negative_probability = sigmoid(
    np.dot(new_review_negative, weights) + bias
)

logging.info(
    "Negative Review Probability = %.4f",
    negative_probability
)

if negative_probability >= 0.5:
    logging.info(
        "Prediction = Positive Sentiment"
    )
else:
    logging.info(
        "Prediction = Negative Sentiment"
    )