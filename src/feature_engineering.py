"""Feature engineering and the preprocessing pipeline.

Two kinds of transformation live here:

* ``engineer_features`` - row-by-row features computed from a customer's own
  values with FIXED rules (e.g. tenure bins at 12/24/48 months). Nothing is
  learned from the dataset, so applying it before or after the split gives
  identical results and cannot leak information.

* ``build_preprocessor`` - imputation, scaling and one-hot encoding. These
  steps DO learn from data (medians, means, category lists), so they live
  inside a scikit-learn Pipeline and are fitted on the training split only.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SERVICE_COLUMNS

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
AGE_BINS = [0, 29, 44, 59, np.inf]
AGE_LABELS = ["18-29", "30-44", "45-59", "60+"]

ENGINEERED_NUMERIC = [
    "service_count",
    "avg_monthly_spend",
    "charge_increase_ratio",
    "is_new_customer",
    "has_streaming",
    "has_protection_bundle",
]
ENGINEERED_CATEGORICAL = ["tenure_group", "age_group"]

MODEL_NUMERIC_FEATURES = NUMERIC_COLUMNS + ENGINEERED_NUMERIC
MODEL_CATEGORICAL_FEATURES = CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL

FEATURE_DESCRIPTIONS = {
    "service_count": "Number of subscribed services (phone + add-ons)",
    "avg_monthly_spend": "Total charges / tenure - historical average bill",
    "charge_increase_ratio": "Current monthly charge / historical average bill",
    "is_new_customer": "1 if tenure is 6 months or less",
    "has_streaming": "1 if the customer streams TV or movies",
    "has_protection_bundle": "1 if the customer has both online security and tech support",
    "tenure_group": "Tenure bucket (0-12, 13-24, 25-48, 49+ months)",
    "age_group": "Age bucket (18-29, 30-44, 45-59, 60+)",
}


def engineer_features(df):
    """Add engineered features. Works on a copy; original columns are kept."""
    df = df.copy()
    yes = {c: (df[c] == "Yes").astype(float) if c in df else 0.0 for c in SERVICE_COLUMNS}

    df["service_count"] = sum(yes.values())
    if "internet_service" in df:
        # Internet itself counts as a service.
        df["service_count"] = df["service_count"] + df["internet_service"].isin(
            ["DSL", "Fiber optic"]
        ).astype(float)

    tenure = pd.to_numeric(df["tenure_months"], errors="coerce")
    total = pd.to_numeric(df["total_charges"], errors="coerce")
    monthly = pd.to_numeric(df["monthly_charges"], errors="coerce")

    df["avg_monthly_spend"] = np.where(tenure > 0, total / tenure.clip(lower=1), monthly)
    ratio = monthly / pd.Series(df["avg_monthly_spend"], index=df.index).replace(0, np.nan)
    df["charge_increase_ratio"] = ratio.clip(0, 5)
    df["is_new_customer"] = np.where(tenure.isna(), np.nan, (tenure <= 6).astype(float))
    df["has_streaming"] = ((yes["streaming_tv"] + yes["streaming_movies"]) > 0).astype(float)
    df["has_protection_bundle"] = (yes["online_security"] * yes["tech_support"]).astype(float)

    df["tenure_group"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(object)
    age = pd.to_numeric(df["age"], errors="coerce")
    df["age_group"] = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS).astype(object)
    # pd.cut returns NaN for missing inputs; keep them as real missing values.
    for col in ENGINEERED_CATEGORICAL:
        df[col] = df[col].where(df[col].notna(), np.nan)
    return df


def select_model_features(df):
    """Keep only the columns the model uses (drops ID, date and target)."""
    return df[MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES]


def _engineer_and_select(df):
    return select_model_features(engineer_features(df))


def build_preprocessor(scale_numeric=True):
    """ColumnTransformer that imputes, scales and one-hot encodes features."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), MODEL_NUMERIC_FEATURES),
            ("cat", Pipeline(categorical_steps), MODEL_CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=True,
    )


def build_pipeline(model, scale_numeric=True):
    """Full pipeline: raw cleaned data -> engineered features -> model."""
    return Pipeline(
        [
            ("features", FunctionTransformer(_engineer_and_select, validate=False)),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def original_feature_name(transformed_name):
    """Map a transformed column (e.g. 'cat__contract_type_Two year') back to
    the original feature ('contract_type')."""
    prefix, _, rest = transformed_name.partition("__")
    if prefix == "num":
        return rest
    # Longest matching categorical name wins (avoids prefix collisions).
    for col in sorted(MODEL_CATEGORICAL_FEATURES, key=len, reverse=True):
        if rest == col or rest.startswith(col + "_"):
            return col
    return rest
