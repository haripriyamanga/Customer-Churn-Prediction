"""Tests for data loading, cleaning, validation and feature engineering."""

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_VALUES, DATA_PATH, REQUIRED_INPUT_COLUMNS
from src.data_processing import (
    clean_data,
    encode_target,
    load_raw_data,
    validate_input_data,
)
from src.feature_engineering import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    build_preprocessor,
    engineer_features,
    original_feature_name,
    select_model_features,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_project_dataset_loads_with_expected_columns():
    raw = load_raw_data(DATA_PATH)
    assert len(raw) > 1000
    for col in REQUIRED_INPUT_COLUMNS + ["customer_id", "churn"]:
        assert col in raw.columns


def test_project_dataset_cleans_to_valid_categories():
    df, report = clean_data(load_raw_data(DATA_PATH))
    assert report.rows_out == len(df)
    assert set(df["churn"].unique()) == {0, 1}
    assert not df.duplicated().any()
    assert df["customer_id"].is_unique
    for col, allowed in CATEGORICAL_VALUES.items():
        assert set(df[col].dropna().unique()) <= set(allowed), col


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def test_duplicates_are_removed(messy_raw):
    df, report = clean_data(messy_raw)
    assert report.duplicate_rows_removed == 1
    assert df["customer_id"].is_unique


def test_inconsistent_categories_are_standardised(messy_raw):
    df, report = clean_data(messy_raw, require_target=False)
    by_id = df.set_index("customer_id")
    assert by_id.loc["C1", "gender"] == "Female"
    assert by_id.loc["C3", "gender"] == "Female"
    assert by_id.loc["C1", "contract_type"] == "Month-to-month"
    assert by_id.loc["C3", "contract_type"] == "Month-to-month"
    assert by_id.loc["C1", "internet_service"] == "Fiber optic"
    assert by_id.loc["C4", "internet_service"] == "Fiber optic"
    assert by_id.loc["C3", "payment_method"] == "Electronic check"
    assert report.categories_standardised["gender"] >= 3


def test_unknown_categories_become_missing_not_guessed(messy_raw):
    df, report = clean_data(messy_raw, require_target=False)
    by_id = df.set_index("customer_id")
    assert pd.isna(by_id.loc["C5", "gender"])
    assert pd.isna(by_id.loc["C5", "contract_type"])
    assert report.unknown_categories_set_missing["contract_type"] == 1


def test_invalid_numbers_become_missing(messy_raw):
    df, report = clean_data(messy_raw, require_target=False)
    by_id = df.set_index("customer_id")
    assert np.isnan(by_id.loc["C2", "age"])              # 150 is out of range
    assert np.isnan(by_id.loc["C3", "monthly_charges"])  # negative charge
    assert np.isnan(by_id.loc["C3", "satisfaction_score"])  # 9 is outside 1-5
    assert np.isnan(by_id.loc["C5", "monthly_charges"])  # "abc"
    assert df["age"].dtype == float
    assert report.invalid_numbers_set_missing["monthly_charges"] == 2


def test_blank_total_charges_with_zero_tenure_set_to_zero(messy_raw):
    df, report = clean_data(messy_raw, require_target=False)
    assert df.set_index("customer_id").loc["C3", "total_charges"] == 0.0
    assert report.total_charges_zero_filled == 1


def test_missing_values_are_kept_for_pipeline_imputation(messy_raw):
    df, _ = clean_data(messy_raw, require_target=False)
    by_id = df.set_index("customer_id")
    # Empty age and empty payment method stay missing (imputed later, on train data only).
    assert np.isnan(by_id.loc["C3", "age"])
    assert pd.isna(by_id.loc["C4", "payment_method"])


def test_rows_with_missing_target_are_removed_and_counted(messy_raw):
    df, report = clean_data(messy_raw, require_target=True)
    assert "C4" not in set(df["customer_id"])
    assert report.missing_target_removed == 1
    assert df["churn"].tolist() == [1, 0, 0, 1]


def test_dates_are_parsed_and_invalid_dates_counted(messy_raw):
    df, report = clean_data(messy_raw, require_target=False)
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])
    assert report.invalid_dates == 1


def test_encode_target_handles_variants():
    out = encode_target(pd.Series(["Yes", "no", "Y", "0", None, "maybe"]))
    assert out.tolist()[:4] == [1, 0, 1, 0]
    assert out.iloc[4:].isna().all()


# ---------------------------------------------------------------------------
# Validation of uploads
# ---------------------------------------------------------------------------
def test_validation_rejects_missing_columns(clean_synthetic):
    bad = clean_synthetic.drop(columns=["contract_type", "age"])
    result = validate_input_data(bad)
    assert result["errors"]
    assert "contract_type" in result["errors"][0]


def test_validation_accepts_good_data(clean_synthetic):
    result = validate_input_data(clean_synthetic.astype(str))
    assert result["errors"] == []


def test_validation_flags_non_numeric_and_unknown_categories(clean_synthetic):
    df = clean_synthetic.astype(str).copy()
    df["age"] = "old"
    df.loc[0, "contract_type"] = "Weekly"
    result = validate_input_data(df)
    assert any("age" in e for e in result["errors"])
    assert any("Weekly" in w for w in result["warnings"])


def test_validation_empty_file():
    assert validate_input_data(pd.DataFrame())["errors"]


# ---------------------------------------------------------------------------
# Feature engineering / preprocessing
# ---------------------------------------------------------------------------
def test_engineered_features(clean_synthetic):
    df, _ = clean_data(clean_synthetic.astype(str))
    eng = engineer_features(df)
    for col in ["service_count", "avg_monthly_spend", "tenure_group", "age_group",
                "is_new_customer", "has_streaming", "charge_increase_ratio"]:
        assert col in eng
    assert eng["service_count"].between(0, 8).all()
    assert set(eng["tenure_group"].dropna()) <= {
        "0-12 months", "13-24 months", "25-48 months", "49+ months"}
    row = eng.iloc[0]
    if row["tenure_months"] > 0:
        assert np.isclose(row["avg_monthly_spend"], row["total_charges"] / row["tenure_months"])


def test_engineer_features_handles_missing_values():
    df = pd.DataFrame([{c: np.nan for c in REQUIRED_INPUT_COLUMNS}])
    eng = engineer_features(df)
    assert len(eng) == 1
    assert pd.isna(eng.loc[0, "tenure_group"])


def test_preprocessor_imputes_and_encodes_without_nans(messy_raw):
    df, _ = clean_data(messy_raw, require_target=False)
    X = select_model_features(engineer_features(df))
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert out.shape[0] == len(df)
    assert not np.isnan(out).any()


def test_preprocessor_learns_only_from_training_rows(clean_synthetic):
    """Imputation statistics must come from the data it is fitted on (train only)."""
    df, _ = clean_data(clean_synthetic.astype(str))
    X = select_model_features(engineer_features(df))
    train, test = X.iloc[:400], X.iloc[400:].copy()
    test["age"] = 99  # extreme values in the "test" set must not affect the fit
    pre = build_preprocessor().fit(train)
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    age_idx = MODEL_NUMERIC_FEATURES.index("age")
    assert imputer.statistics_[age_idx] == train["age"].median()


def test_original_feature_name_mapping():
    assert original_feature_name("num__tenure_months") == "tenure_months"
    assert original_feature_name("cat__contract_type_Two year") == "contract_type"
    assert original_feature_name("cat__tenure_group_0-12 months") == "tenure_group"
    assert set(MODEL_CATEGORICAL_FEATURES) >= {"contract_type", "tenure_group"}
