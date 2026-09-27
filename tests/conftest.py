"""Shared test fixtures. Tests never need an external API or saved model files."""

import pandas as pd
import pytest

from src.generate_data import generate_clean_data


@pytest.fixture(scope="session")
def clean_synthetic():
    """A small tidy synthetic dataset (raw string-like values, no messiness)."""
    return generate_clean_data(n=600, seed=7)


@pytest.fixture
def messy_raw():
    """A tiny hand-written raw dataset containing typical data-quality issues."""
    rows = [
        # customer_id, signup_date, gender, age, tenure, contract, payment, internet,
        # monthly, total, tickets, satisfaction, churn
        ["C1", "2024-01-05", "F", "34", "5", "month-to-month", "Electronic check", "Fibre",
         "80.5", "402.5", "2", "3", "Yes"],
        ["C2", "2020-03-01", "Male", "150", "48", "Two year", "Credit card", "DSL",
         "55", "2640", "0", "5", "No"],
        ["C3", "not a date", "female", "", "0", "MTM", " Electronic Check", "No",
         "-20", " ", "1", "9", "no"],
        ["C4", "2023-07-19", "M", "61", "12", "One year", "", "fiber optic",
         "95.1", "1141.2", "4", "2", ""],
        ["C5", "2022-02-02", "Alien", "45", "30", "Weekly", "Bank transfer", "DSL",
         "abc", "1500", "1", "4", "Y"],
    ]
    cols = ["customer_id", "signup_date", "gender", "age", "tenure_months", "contract_type",
            "payment_method", "internet_service", "monthly_charges", "total_charges",
            "support_tickets", "satisfaction_score", "churn"]
    df = pd.DataFrame(rows, columns=cols)
    fill = {"partner": "Yes", "dependents": "No", "region": "North", "paperless_billing": "Yes",
            "phone_service": "Yes", "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No", "streaming_tv": "Yes",
            "streaming_movies": "No", "late_payments": "0"}
    for col, value in fill.items():
        df[col] = value
    # Exact duplicate of the first row.
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)
