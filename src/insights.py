"""Business insights and rule-based retention recommendations.

Insights are computed from the data and model predictions every time - no
numbers are typed in by hand. Recommendations are SUGGESTED actions based on
simple, transparent rules; they are not guaranteed to prevent churn.
"""

import pandas as pd

from src.config import TARGET_COLUMN
from src.feature_engineering import engineer_features

SEGMENT_COLUMNS = {
    "contract_type": "Contract type",
    "tenure_group": "Tenure group",
    "payment_method": "Payment method",
    "internet_service": "Internet service",
    "age_group": "Age group",
    "region": "Region",
    "tech_support": "Tech support",
}


def churn_rate_by(df, column, min_customers=30):
    """Churn rate, customer count and churned count per category of ``column``."""
    data = engineer_features(df) if column not in df.columns else df
    grouped = (
        data.dropna(subset=[column])
        .groupby(column, observed=True)[TARGET_COLUMN]
        .agg(customers="count", churned="sum", churn_rate="mean")
        .reset_index()
    )
    grouped = grouped[grouped["customers"] >= min_customers]
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def highest_churn_segment(df, columns=("contract_type", "tenure_group", "internet_service"),
                          min_customers=100):
    """Find the combination of two attributes with the highest churn rate."""
    data = engineer_features(df)
    best = None
    cols = list(columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            g = (
                data.dropna(subset=[a, b])
                .groupby([a, b], observed=True)[TARGET_COLUMN]
                .agg(customers="count", churn_rate="mean")
                .reset_index()
            )
            g = g[g["customers"] >= min_customers]
            if g.empty:
                continue
            top = g.sort_values("churn_rate", ascending=False).iloc[0]
            if best is None or top["churn_rate"] > best["churn_rate"]:
                best = {
                    "segment": f"{SEGMENT_COLUMNS.get(a, a)} = {top[a]} & "
                               f"{SEGMENT_COLUMNS.get(b, b)} = {top[b]}",
                    "customers": int(top["customers"]),
                    "churn_rate": float(top["churn_rate"]),
                }
    return best


def generate_insights(df, scored=None, importance=None):
    """Return a list of plain-English insight strings computed from the data.

    ``scored`` - the dataset with a ``risk_category`` column (optional).
    ``importance`` - DataFrame of feature importances (optional).
    """
    insights = []
    overall = df[TARGET_COLUMN].mean()
    insights.append(
        f"The overall churn rate is {overall:.1%} "
        f"({int(df[TARGET_COLUMN].sum()):,} of {len(df):,} customers)."
    )

    for col in ["contract_type", "tenure_group", "payment_method", "internet_service"]:
        rates = churn_rate_by(df, col)
        if len(rates) < 2:
            continue
        top, low = rates.iloc[0], rates.iloc[-1]
        ratio = top["churn_rate"] / low["churn_rate"] if low["churn_rate"] > 0 else float("inf")
        insights.append(
            f"{SEGMENT_COLUMNS[col]}: '{top[col]}' has the highest churn rate "
            f"({top['churn_rate']:.1%}, n={int(top['customers']):,}) versus "
            f"{low['churn_rate']:.1%} for '{low[col]}'"
            + (f" - about {ratio:.1f}x higher." if ratio != float("inf") else ".")
        )

    seg = highest_churn_segment(df)
    if seg:
        insights.append(
            f"The highest-churn segment (min. 100 customers) is {seg['segment']}: "
            f"{seg['churn_rate']:.1%} churn across {seg['customers']:,} customers."
        )

    churned = df[df[TARGET_COLUMN] == 1]["monthly_charges"].mean()
    stayed = df[df[TARGET_COLUMN] == 0]["monthly_charges"].mean()
    insights.append(
        f"Customers who churned paid on average ${churned:,.2f}/month versus "
        f"${stayed:,.2f}/month for customers who stayed."
    )

    if scored is not None and "risk_category" in scored:
        counts = scored["risk_category"].value_counts()
        high = int(counts.get("High Risk", 0))
        insights.append(
            f"The primary model classifies {high:,} customers ({high / len(scored):.1%}) as "
            f"High Risk, {int(counts.get('Medium Risk', 0)):,} as Medium Risk and "
            f"{int(counts.get('Low Risk', 0)):,} as Low Risk."
        )

    if importance is not None and len(importance):
        top = ", ".join(importance["feature"].head(5))
        insights.append(
            f"The features with the strongest influence on the model's predictions are: {top}. "
            "(Influence on predictions is an association, not proof of cause.)"
        )
    return insights


# ---------------------------------------------------------------------------
# Rule-based retention recommendations
# ---------------------------------------------------------------------------
def retention_recommendations(customer, probability, risk, reference, drivers=None):
    """Suggest retention actions for one customer.

    Parameters
    ----------
    customer : dict-like of the customer's cleaned values.
    probability : predicted churn probability.
    risk : risk category string.
    reference : dict with dataset reference values, e.g.
        ``{"high_monthly_charges": <75th percentile of training data>}``.
    drivers : optional set of features that SHAP says INCREASE this
        customer's churn risk - used to flag which recommendations are backed
        by the model's own explanation.

    Returns a list of dicts: {"action", "reason", "supported_by_model"}.
    """
    drivers = set(drivers or [])
    recs = []

    def add(action, reason, related):
        recs.append(
            {
                "action": action,
                "reason": reason,
                "supported_by_model": bool(drivers & set(related)),
            }
        )

    if risk == "Low Risk":
        return [{
            "action": "No targeted intervention needed - continue standard customer care.",
            "reason": f"Predicted churn probability is low ({probability:.0%}).",
            "supported_by_model": False,
        }]

    tenure = customer.get("tenure_months")
    monthly = customer.get("monthly_charges")
    if pd.notna(tenure) and tenure <= 12:
        add("Early customer engagement: onboarding check-in, welcome offer or usage tips.",
            f"Short tenure ({int(tenure)} months) combined with elevated churn risk.",
            ["tenure_months", "tenure_group", "is_new_customer", "avg_monthly_spend"])
    high_charge = reference.get("high_monthly_charges")
    if pd.notna(monthly) and high_charge is not None and monthly >= high_charge:
        add("Review pricing / value: check the plan fits their usage or offer a better-value bundle.",
            f"Monthly charges (${monthly:,.2f}) are in the top 25% of customers "
            f"(>= ${high_charge:,.2f}).",
            ["monthly_charges", "charge_increase_ratio", "avg_monthly_spend"])
    tickets = customer.get("support_tickets")
    satisfaction = customer.get("satisfaction_score")
    if (pd.notna(tickets) and tickets >= 3) or (pd.notna(satisfaction) and satisfaction <= 2):
        detail = []
        if pd.notna(tickets) and tickets >= 3:
            detail.append(f"{int(tickets)} support tickets")
        if pd.notna(satisfaction) and satisfaction <= 2:
            detail.append(f"satisfaction score {int(satisfaction)}/5")
        add("Support follow-up: proactive call from a senior agent to resolve open issues.",
            "Service-related signals: " + " and ".join(detail) + ".",
            ["support_tickets", "satisfaction_score"])
    if customer.get("contract_type") == "Month-to-month":
        add("Offer an incentive to move to a 1- or 2-year contract.",
            "Month-to-month contract - the customer can leave at any time.",
            ["contract_type"])
    if customer.get("payment_method") == "Electronic check" or (
        pd.notna(customer.get("late_payments")) and customer.get("late_payments", 0) >= 2
    ):
        add("Encourage automatic payment (bank transfer / credit card) and send payment reminders.",
            "Manual electronic-check payments and/or repeated late payments.",
            ["payment_method", "late_payments"])
    if customer.get("internet_service") in ("DSL", "Fiber optic") and (
        customer.get("tech_support") == "No" or customer.get("online_security") == "No"
    ):
        add("Offer a free trial of tech support / online security.",
            "Internet customer without tech support and/or online security add-ons.",
            ["tech_support", "online_security", "has_protection_bundle"])

    if not recs:
        add("Monitor the account and review it at the next billing cycle.",
            "Elevated risk but none of the specific rule conditions apply.", [])
    # Recommendations backed by the model's explanation first.
    return sorted(recs, key=lambda r: not r["supported_by_model"])
