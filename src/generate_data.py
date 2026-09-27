"""Generate the SYNTHETIC customer churn dataset used by this project.

Every record is fictional. Customer IDs are sequential labels, and no name,
email, phone number or address is created. The data follows plausible
telecom-style patterns (for example, month-to-month contracts and short
tenure are associated with higher churn) so the modelling workflow has real
signal to learn, but it does NOT describe any real company or person.

A small amount of "messiness" is injected on purpose (missing values,
duplicate rows, inconsistent category spellings, invalid numbers, blank
strings) so the data-cleaning step has something realistic to do.

Usage:
    python src/generate_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PATH, RANDOM_STATE  # noqa: E402

N_CUSTOMERS = 7_000
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")


def _yes_no(rng, p_yes, size=None):
    return np.where(rng.random(size) < p_yes, "Yes", "No")


def generate_clean_data(n=N_CUSTOMERS, seed=RANDOM_STATE):
    """Create a tidy synthetic dataset (before messiness is injected)."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(44, 15, n).round(), 18, 85).astype(int)
    gender = rng.choice(["Female", "Male"], n)
    partner = _yes_no(rng, 0.48, n)
    dependents = np.where(partner == "Yes", _yes_no(rng, 0.45, n), _yes_no(rng, 0.12, n))
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.27, 0.25, 0.26, 0.22])

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]
    )
    # Longer contracts tend to belong to longer-tenure customers.
    tenure_mean = np.select(
        [contract_type == "Month-to-month", contract_type == "One year"], [16, 36], 52
    )
    tenure = np.clip(rng.gamma(2.0, tenure_mean / 2.0), 0, 72).round().astype(int)

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21])
    has_internet = internet_service != "No"
    phone_service = _yes_no(rng, 0.90, n)

    def addon(p):
        return np.where(has_internet, _yes_no(rng, p, n), "No")

    online_security = addon(0.35)
    online_backup = addon(0.42)
    device_protection = addon(0.42)
    tech_support = addon(0.36)
    streaming_tv = addon(0.50)
    streaming_movies = addon(0.50)

    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n,
        p=[0.34, 0.22, 0.22, 0.22],
    )
    paperless_billing = _yes_no(rng, 0.59, n)

    # Monthly price built from the services a customer has, plus noise.
    monthly = (
        np.where(phone_service == "Yes", 20.0, 0.0)
        + np.select([internet_service == "DSL", internet_service == "Fiber optic"], [25.0, 45.0], 0.0)
        + 5.0 * (online_security == "Yes")
        + 5.0 * (online_backup == "Yes")
        + 5.0 * (device_protection == "Yes")
        + 5.0 * (tech_support == "Yes")
        + 10.0 * (streaming_tv == "Yes")
        + 10.0 * (streaming_movies == "Yes")
        + rng.normal(0, 4, n)
    )
    monthly = np.clip(monthly, 18.0, 130.0).round(2)
    total = (monthly * tenure * rng.normal(1.0, 0.04, n)).round(2)
    total = np.where(tenure == 0, 0.0, total)

    support_tickets = rng.poisson(
        0.8 + 1.2 * (internet_service == "Fiber optic") + 0.8 * (tech_support == "No") * has_internet
    )
    satisfaction = np.clip(
        np.round(rng.normal(3.7 - 0.25 * support_tickets, 0.9)), 1, 5
    ).astype(int)
    late_payments = rng.poisson(0.4 + 0.6 * (payment_method == "Electronic check"))

    # ---- churn: a logistic function of the attributes above --------------
    logit = (
        -1.8
        + 1.30 * (contract_type == "Month-to-month")
        - 0.90 * (contract_type == "Two year")
        - 0.045 * tenure
        + 0.60 * (internet_service == "Fiber optic")
        + 0.018 * (monthly - 65)
        + 0.45 * (payment_method == "Electronic check")
        - 0.45 * (online_security == "Yes")
        - 0.50 * (tech_support == "Yes")
        + 0.22 * support_tickets
        - 0.45 * (satisfaction - 3)
        + 0.20 * late_payments
        + 0.25 * (paperless_billing == "Yes")
        - 0.20 * (partner == "Yes")
        + 0.012 * np.clip(age - 60, 0, None) * 2
        + rng.normal(0, 0.35, n)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    signup_date = SNAPSHOT_DATE - pd.to_timedelta(tenure * 30 + rng.integers(0, 29, n), unit="D")

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "gender": gender,
            "age": age,
            "partner": partner,
            "dependents": dependents,
            "region": region,
            "tenure_months": tenure,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "internet_service": internet_service,
            "phone_service": phone_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_tickets": support_tickets,
            "satisfaction_score": satisfaction,
            "late_payments": late_payments,
            "churn": churn,
        }
    )


def inject_messiness(df, seed=RANDOM_STATE):
    """Add realistic data-quality problems so cleaning can be demonstrated."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy().astype(object)
    n = len(df)

    def pick(frac):
        return rng.choice(n, size=int(n * frac), replace=False)

    # 1) Missing values in several columns (~1-2%).
    for col, frac in [
        ("age", 0.015),
        ("monthly_charges", 0.01),
        ("payment_method", 0.01),
        ("internet_service", 0.005),
        ("satisfaction_score", 0.02),
        ("support_tickets", 0.01),
    ]:
        df.loc[pick(frac), col] = np.nan

    # 2) Total charges stored as text with blanks for some customers
    #    (a classic issue in exported billing data).
    df["total_charges"] = df["total_charges"].astype(str)
    df.loc[df["tenure_months"] == 0, "total_charges"] = " "
    df.loc[pick(0.003), "total_charges"] = ""

    # 3) Inconsistent category spellings / casing / whitespace.
    messy_maps = {
        "contract_type": {"Month-to-month": ["month-to-month", "Month to Month", "MTM"]},
        "internet_service": {"Fiber optic": ["fiber optic", "Fiber Optic ", "Fibre"]},
        "payment_method": {"Electronic check": ["electronic check", " Electronic Check"]},
        "gender": {"Female": ["F", "female"], "Male": ["M", "male"]},
        "partner": {"Yes": ["Y", "yes"], "No": ["N", "no"]},
    }
    for col, mapping in messy_maps.items():
        for canonical, variants in mapping.items():
            idx = df.index[df[col] == canonical]
            chosen = rng.choice(idx, size=max(1, int(len(idx) * 0.03)), replace=False)
            df.loc[chosen, col] = rng.choice(variants, size=len(chosen))

    # 4) Invalid numeric values (data-entry errors).
    df.loc[pick(0.002), "age"] = rng.choice([-1, 0, 150, 999], size=int(n * 0.002))
    df.loc[pick(0.002), "monthly_charges"] = -rng.uniform(10, 50, int(n * 0.002)).round(2)
    df.loc[pick(0.001), "satisfaction_score"] = 9

    # 5) Unparseable dates.
    df.loc[pick(0.002), "signup_date"] = "not a date"

    # 6) A few rows with an unknown target label.
    df.loc[pick(0.002), "churn"] = np.nan

    # 7) Exact duplicate rows (e.g. from a double export).
    dupes = df.loc[pick(0.006)]
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    df = inject_messiness(generate_clean_data())
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    churn_rate = (df["churn"] == "Yes").mean()
    print(f"Wrote {len(df):,} rows to {DATA_PATH} (raw churn rate ~{churn_rate:.1%}).")


if __name__ == "__main__":
    main()
