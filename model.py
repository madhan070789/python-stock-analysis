"""
model.py
========
This module handles the machine learning side of crash prediction.

It uses a Random Forest Classifier trained on synthetic data to predict
whether a market crash is likely based on four features:
  1. Price change (%)
  2. Volume change (%)
  3. Volatility (%)
  4. Sentiment score (-1 to +1)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generate_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic training data for the crash prediction model.

    The logic:
    - A "crash" label (1) is assigned when conditions look bad:
        * Large negative price change
        * Spike in volume (panic selling)
        * High volatility
        * Negative sentiment
    - A "no crash" label (0) is assigned for normal / calm conditions.

    Parameters
    ----------
    n_samples : int
        Number of synthetic samples to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: price_change, volume_change, volatility,
        sentiment_score, crash (0 or 1).
    """
    np.random.seed(42)  # For reproducibility

    data = []

    for _ in range(n_samples):
        # Randomly decide if this sample represents a crash scenario
        is_crash = np.random.random() < 0.3  # 30% crash samples

        if is_crash:
            # Crash scenario: bad numbers, but with some noise/overlap
            price_change = np.random.normal(-4, 3)           # Mean -4%, std 3%
            volume_change = np.random.normal(50, 40)         # Mean 50%, std 40%
            volatility = np.random.normal(5, 2)              # Mean 5%, std 2%
            sentiment = np.random.normal(-0.4, 0.3)          # Mean -0.4, std 0.3
            label = 1
        else:
            # Normal scenario: stable or positive numbers, with overlap
            price_change = np.random.normal(1, 2)            # Mean 1%, std 2%
            volume_change = np.random.normal(0, 20)          # Mean 0%, std 20%
            volatility = np.random.normal(2, 1)              # Mean 2%, std 1%
            sentiment = np.random.normal(0.2, 0.4)           # Mean 0.2, std 0.4
            label = 0

        data.append({
            "price_change": round(price_change, 4),
            "volume_change": round(volume_change, 4),
            "volatility": round(volatility, 4),
            "sentiment_score": round(sentiment, 4),
            "crash": label,
        })

    return pd.DataFrame(data)


def train_model() -> tuple:
    """
    Train a Random Forest Classifier on synthetic data.

    Returns
    -------
    tuple
        (trained_model, accuracy_score)
        - trained_model : fitted RandomForestClassifier
        - accuracy_score : float, accuracy on the test set (0 to 1)
    """
    # Step 1: Generate training data
    df = generate_training_data(n_samples=2000)

    # Step 2: Split into features (X) and labels (y)
    X = df[["price_change", "volume_change", "volatility", "sentiment_score"]]
    y = df["crash"]

    # Step 3: Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Maximum depth of each tree
        random_state=42,       # Reproducibility
    )
    model.fit(X_train, y_train)

    # Step 5: Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def predict_crash(model, features: dict) -> dict:
    """
    Predict crash probability using the trained model.

    Parameters
    ----------
    model : RandomForestClassifier
        The trained model from train_model().
    features : dict
        Dictionary with keys: 'price_change', 'volume_change',
        'volatility', 'sentiment_score'.

    Returns
    -------
    dict
        Dictionary with:
        - 'crash_probability' : float (0 to 100, percentage)
        - 'risk_level'        : str ('LOW', 'MEDIUM', 'HIGH', or 'CRITICAL')
        - 'prediction'        : str ('Crash Likely' or 'Market Stable')
    """
    # Build the input array in the correct feature order
    input_data = np.array([[
        features["price_change"],
        features["volume_change"],
        features["volatility"],
        features["sentiment_score"],
    ]])

    # Get crash probability (probability of class 1 = crash)
    probabilities = model.predict_proba(input_data)
    crash_prob = probabilities[0][1] * 100  # Convert to percentage

    # Determine risk level based on probability
    if crash_prob < 25:
        risk_level = "LOW"
    elif crash_prob < 50:
        risk_level = "MEDIUM"
    elif crash_prob < 75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    # Determine prediction text
    prediction = "Crash Likely" if crash_prob >= 50 else "Market Stable"

    return {
        "crash_probability": round(crash_prob, 2),
        "risk_level": risk_level,
        "prediction": prediction,
    }
