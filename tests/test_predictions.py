"""Tests for the prediction pipeline, risk categories, evaluation, explanations
and recommendations. A small model is trained in-memory so the tests do not
depend on saved model files."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import HIGH_RISK_MIN, LOW_RISK_MAX
from src.data_processing import clean_data
from src.evaluate_models import comparison_table, evaluate_predictions, select_primary_model
from src.explainability import SHAP_AVAILABLE, explain_customer, model_feature_importance
from src.feature_engineering import build_pipeline
from src.insights import churn_rate_by, generate_insights, retention_recommendations
from src.predictions import predict_single, risk_category, score_customers


@pytest.fixture(scope="module")
def data(clean_synthetic):
    df, _ = clean_data(clean_synthetic.astype(str))
    return df.drop(columns=["churn"]), df["churn"]


@pytest.fixture(scope="module")
def fitted_lr(data):
    X, y = data
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)


@pytest.fixture(scope="module")
def fitted_xgb(data):
    X, y = data
    model = XGBClassifier(n_estimators=40, max_depth=3, random_state=0, n_jobs=1)
    return build_pipeline(model).fit(X, y)


SAMPLE_CUSTOMER = {
    "gender": "Female", "age": 29, "partner": "No", "dependents": "No", "region": "West",
    "tenure_months": 3, "contract_type": "Month-to-month", "payment_method": "Electronic check",
    "paperless_billing": "Yes", "internet_service": "Fiber optic", "phone_service": "Yes",
    "online_security": "No", "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "Yes",
    "monthly_charges": 99.5, "total_charges": 298.5, "support_tickets": 4,
    "satisfaction_score": 2, "late_payments": 2,
}


# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prob, expected",
    [(0.0, "Low Risk"), (LOW_RISK_MAX - 0.001, "Low Risk"), (LOW_RISK_MAX, "Medium Risk"),
     (HIGH_RISK_MIN - 0.001, "Medium Risk"), (HIGH_RISK_MIN, "High Risk"), (1.0, "High Risk"),
     (float("nan"), "Unknown")],
)
def test_risk_category_thresholds(prob, expected):
    assert risk_category(prob) == expected


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------
def test_pipeline_predicts_valid_probabilities(fitted_lr, data):
    X, _ = data
    proba = fitted_lr.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_score_customers_adds_columns(fitted_lr, data):
    X, _ = data
    scored = score_customers(fitted_lr, X.head(20), threshold=0.4)
    for col in ["churn_probability", "predicted_churn", "risk_category"]:
        assert col in scored
    flagged = scored["churn_probability"] >= 0.4
    assert (scored.loc[flagged, "predicted_churn"] == "Likely to churn").all()


def test_predict_single_from_raw_form_values(fitted_xgb):
    out = predict_single(fitted_xgb, SAMPLE_CUSTOMER)
    assert 0 <= out["probability"] <= 1
    assert out["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert out["prediction"] in {"Likely to churn", "Likely to stay"}


def test_prediction_handles_missing_and_messy_values(fitted_lr):
    customer = dict(SAMPLE_CUSTOMER, age=None, payment_method="", monthly_charges="n/a",
                    contract_type="month to month", internet_service="Fibre")
    out = predict_single(fitted_lr, customer)
    assert 0 <= out["probability"] <= 1
    assert out["cleaned_input"].loc[0, "contract_type"] == "Month-to-month"


def test_high_risk_profile_scores_higher_than_loyal_profile(fitted_lr):
    loyal = dict(SAMPLE_CUSTOMER, tenure_months=70, contract_type="Two year",
                 payment_method="Credit card", support_tickets=0, satisfaction_score=5,
                 total_charges=99.5 * 70, online_security="Yes", tech_support="Yes",
                 late_payments=0)
    assert predict_single(fitted_lr, SAMPLE_CUSTOMER)["probability"] > \
        predict_single(fitted_lr, loyal)["probability"]


def test_unknown_category_at_prediction_time_does_not_crash(fitted_lr):
    out = predict_single(fitted_lr, dict(SAMPLE_CUSTOMER, region="Atlantis"))
    assert 0 <= out["probability"] <= 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def test_evaluate_predictions_metrics():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    res = evaluate_predictions(y, proba, threshold=0.5)
    assert res["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert res["Accuracy"] == 0.5
    assert res["Precision"] == 0.5 and res["Recall"] == 0.5
    assert res["ROC-AUC"] == 0.75


def test_select_primary_model_uses_cv_roc_auc_not_accuracy():
    results = {
        "A": {"cv": {"roc_auc_mean": 0.80, "f1_mean": 0.6}, "test": {"Accuracy": 0.95}},
        "B": {"cv": {"roc_auc_mean": 0.86, "f1_mean": 0.6}, "test": {"Accuracy": 0.70}},
    }
    name, reason = select_primary_model(results, churn_rate=0.25)
    assert name == "B"
    assert "ROC-AUC" in reason


def test_comparison_table_is_dynamic():
    res = evaluate_predictions([0, 1, 1, 0], [0.2, 0.7, 0.8, 0.3])
    table = comparison_table({"M": {"test": res, "cv": {"roc_auc_mean": 0.9, "roc_auc_std": 0.01}}})
    assert table.loc["M", "ROC-AUC"] == 1.0
    assert list(table.columns[:5]) == ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def test_feature_importance_aggregates_to_original_features(fitted_xgb, fitted_lr):
    for pipe in (fitted_xgb, fitted_lr):
        imp, _ = model_feature_importance(pipe)
        assert "contract_type" in set(imp["feature"])
        assert not any(f.startswith(("num__", "cat__")) for f in imp["feature"])
        assert np.isclose(imp["importance"].sum(), 1.0)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap not installed")
def test_local_shap_explanation(fitted_xgb, data):
    X, _ = data
    customer = X.iloc[[0]]
    expl, base = explain_customer(fitted_xgb, customer, X.sample(100, random_state=0))
    assert {"feature", "customer_value", "contribution", "direction"} <= set(expl.columns)
    assert len(expl) > 0
    # SHAP additivity: base + sum of all contributions == model log-odds output.
    full, _ = explain_customer(fitted_xgb, customer, X, top_n=1000)
    margin = fitted_xgb.named_steps["model"].predict(
        fitted_xgb[:-1].transform(customer), output_margin=True)[0]
    assert np.isclose(base + full["contribution"].sum(), margin, atol=1e-3)


# ---------------------------------------------------------------------------
# Insights and recommendations
# ---------------------------------------------------------------------------
def test_churn_rate_by_matches_manual_calculation(clean_synthetic):
    df, _ = clean_data(clean_synthetic.astype(str))
    rates = churn_rate_by(df, "contract_type", min_customers=1).set_index("contract_type")
    manual = df.groupby("contract_type")["churn"].mean()
    for k, v in manual.items():
        assert np.isclose(rates.loc[k, "churn_rate"], v)


def test_generate_insights_returns_computed_text(clean_synthetic):
    df, _ = clean_data(clean_synthetic.astype(str))
    insights = generate_insights(df)
    assert insights and f"{df['churn'].mean():.1%}" in insights[0]


def test_recommendations_follow_rules():
    ref = {"high_monthly_charges": 90.0}
    recs = retention_recommendations(SAMPLE_CUSTOMER, 0.8, "High Risk", ref,
                                     drivers={"tenure_months"})
    actions = " ".join(r["action"] for r in recs)
    assert "Early customer engagement" in actions
    assert "pricing" in actions
    assert "Support follow-up" in actions
    assert recs[0]["supported_by_model"] is True  # SHAP-backed first

    low = retention_recommendations(SAMPLE_CUSTOMER, 0.1, "Low Risk", ref)
    assert len(low) == 1 and "No targeted intervention" in low[0]["action"]


def test_recommendations_handle_missing_values():
    customer = {k: np.nan for k in SAMPLE_CUSTOMER}
    recs = retention_recommendations(customer, 0.7, "High Risk", {"high_monthly_charges": 90.0})
    assert recs  # falls back to a monitoring suggestion
    assert isinstance(pd.DataFrame(recs), pd.DataFrame)
