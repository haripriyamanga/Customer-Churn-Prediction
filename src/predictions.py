"""Loading saved models, scoring customers and assigning risk categories."""

import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALL_MODELS_PATH,
    CLASSIFICATION_THRESHOLD,
    EVALUATION_PATH,
    HIGH_RISK_MIN,
    LOW_RISK_MAX,
    PRIMARY_MODEL_PATH,
)
from src.data_processing import clean_data

RISK_LEVELS = ["Low Risk", "Medium Risk", "High Risk"]


class ModelNotTrainedError(FileNotFoundError):
    """Raised when saved model files are missing."""


def risk_category(probability, low_max=LOW_RISK_MAX, high_min=HIGH_RISK_MIN):
    """Map a churn probability to Low / Medium / High Risk.

    * Low Risk:    probability <  0.25
    * Medium Risk: 0.25 <= probability < 0.60
    * High Risk:   probability >= 0.60
    """
    if probability is None or (isinstance(probability, float) and np.isnan(probability)):
        return "Unknown"
    if probability >= high_min:
        return "High Risk"
    if probability >= low_max:
        return "Medium Risk"
    return "Low Risk"


def load_primary_model(path=PRIMARY_MODEL_PATH):
    """Load the saved primary model bundle (pipeline + metadata)."""
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {path}. Run `python src/train_models.py` first."
        )
    return joblib.load(path)


def load_all_models(path=ALL_MODELS_PATH):
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained models found at {path}. Run `python src/train_models.py` first."
        )
    return joblib.load(path)


def load_evaluation(path=EVALUATION_PATH):
    if not path.exists():
        raise ModelNotTrainedError(
            f"No evaluation results found at {path}. Run `python src/train_models.py` first."
        )
    return json.loads(path.read_text())


def predict_proba(pipeline, df):
    """Churn probability (class 1) for each row of cleaned data."""
    return pipeline.predict_proba(df)[:, 1]


def score_customers(pipeline, df, threshold=CLASSIFICATION_THRESHOLD):
    """Add churn probability, predicted label and risk category columns."""
    out = df.copy()
    proba = predict_proba(pipeline, df)
    out["churn_probability"] = proba
    out["predicted_churn"] = np.where(proba >= threshold, "Likely to churn", "Likely to stay")
    out["risk_category"] = [risk_category(p) for p in proba]
    return out


def predict_single(pipeline, customer, threshold=CLASSIFICATION_THRESHOLD):
    """Score one customer given as a dict of raw values.

    The dict goes through the same cleaning as the training data, so the
    form can pass plain strings/numbers.
    """
    raw = pd.DataFrame([customer]).astype(object)
    cleaned, _ = clean_data(raw, require_target=False)
    proba = float(predict_proba(pipeline, cleaned)[0])
    return {
        "probability": proba,
        "prediction": "Likely to churn" if proba >= threshold else "Likely to stay",
        "risk_category": risk_category(proba),
        "cleaned_input": cleaned,
    }
