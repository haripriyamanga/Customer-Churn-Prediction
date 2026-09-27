"""Explainable AI helpers: feature importance and SHAP explanations.

Important caveat (shown in the dashboard as well): these methods describe
which inputs most influence the MODEL's output. They do not prove that a
feature causes churn in the real world.

All one-hot encoded columns are aggregated back to their original feature
(e.g. the three ``contract_type_*`` columns become ``contract_type``). This is
valid for SHAP values because SHAP contributions are additive.
"""

import numpy as np
import pandas as pd

from src.feature_engineering import original_feature_name

try:  # SHAP is optional - the app degrades gracefully without it.
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    shap = None
    SHAP_AVAILABLE = False


def _split_pipeline(pipeline):
    """Return (transform_fn, feature_names, model) for a fitted pipeline."""
    transformer = pipeline[:-1]
    model = pipeline.named_steps["model"]
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    return transformer.transform, names, model


def _aggregate(values, names):
    """Sum values of transformed columns that belong to the same original feature."""
    series = pd.Series(values, index=[original_feature_name(n) for n in names])
    return series.groupby(level=0).sum()


def model_feature_importance(pipeline):
    """Model-native global importance, aggregated to original features.

    * Tree models: impurity-based ``feature_importances_``.
    * Logistic Regression: absolute coefficients (features are standardised,
      so coefficient sizes are comparable).
    Returns a DataFrame sorted by importance (normalised to sum to 1).
    """
    _, names, model = _split_pipeline(pipeline)
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
        method = "Impurity / gain-based importance"
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        method = "Absolute standardised coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance"]), "Not available"
    agg = _aggregate(raw, names)
    total = agg.sum()
    if total > 0:
        agg = agg / total
    df = agg.sort_values(ascending=False).rename("importance").reset_index()
    df.columns = ["feature", "importance"]
    return df, method


def _shap_explainer(model, background):
    """Pick an appropriate SHAP explainer for the model type."""
    name = type(model).__name__
    if name in {"XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier"}:
        return shap.TreeExplainer(model)
    if name == "LogisticRegression":
        return shap.LinearExplainer(model, background)
    return shap.Explainer(model.predict_proba, background)


def _positive_class_values(explanation_values):
    """Normalise SHAP output to shape (n_samples, n_features) for class 1."""
    values = np.asarray(explanation_values)
    if isinstance(explanation_values, list):  # older SHAP: list per class
        values = np.asarray(explanation_values[1])
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, 1]
    return values


def compute_shap_values(pipeline, X, background=None, max_background=200, random_state=42):
    """Compute SHAP values (class = churn) for rows of cleaned data ``X``.

    Returns a DataFrame of per-feature SHAP values aggregated to original
    features, plus the explainer's base value (in the model's output space,
    log-odds for XGBoost/Logistic Regression, probability for sklearn trees).
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not installed. Run: pip install shap")
    transform, names, model = _split_pipeline(pipeline)
    Xt = transform(X)
    if background is None:
        background = X
    bg = transform(background)
    if len(bg) > max_background:
        rng = np.random.default_rng(random_state)
        bg = bg[rng.choice(len(bg), max_background, replace=False)]

    explainer = _shap_explainer(model, bg)
    raw = explainer.shap_values(Xt) if hasattr(explainer, "shap_values") else explainer(Xt).values
    values = _positive_class_values(raw)

    base = explainer.expected_value
    base = np.asarray(base).ravel()
    base_value = float(base[1] if len(base) > 1 else base[0])

    per_col = pd.DataFrame(values, columns=names, index=X.index)
    grouped = per_col.T.groupby([original_feature_name(n) for n in names]).sum().T
    return grouped, base_value


def global_shap_importance(pipeline, X, **kwargs):
    """Mean absolute SHAP value per original feature (global importance)."""
    shap_df, _ = compute_shap_values(pipeline, X, **kwargs)
    imp = shap_df.abs().mean().sort_values(ascending=False)
    return imp.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})


def _format_value(value):
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "missing"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def explain_customer(pipeline, customer, background, top_n=8):
    """Local explanation for one customer (a single-row cleaned DataFrame).

    Returns a DataFrame with the feature, the customer's value, the SHAP
    contribution and its direction - sorted by absolute contribution. Only
    factors actually produced by SHAP are returned; nothing is invented.
    """
    from src.feature_engineering import engineer_features

    shap_df, base_value = compute_shap_values(pipeline, customer, background=background)
    contrib = shap_df.iloc[0]
    engineered = engineer_features(customer).iloc[0]
    rows = []
    for feature, value in contrib.items():
        rows.append(
            {
                "feature": feature,
                "customer_value": _format_value(engineered.get(feature, np.nan)),
                "contribution": float(value),
                "direction": "Increases churn risk" if value > 0 else "Decreases churn risk",
            }
        )
    out = pd.DataFrame(rows)
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
    out = out[out["contribution"].abs() > 1e-9]
    return out.head(top_n).reset_index(drop=True), base_value


def shap_output_units(pipeline):
    """Units of the SHAP values for this model type (for chart labels)."""
    name = type(pipeline.named_steps["model"]).__name__
    if name in {"RandomForestClassifier", "DecisionTreeClassifier"}:
        return "probability"
    return "log-odds"
