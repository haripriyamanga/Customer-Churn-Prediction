"""Build (and optionally execute) notebooks/exploratory_analysis.ipynb.

The notebook is generated from this script so its content is easy to review
in code. Run:
    python notebooks/build_notebook.py            # build + execute
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "exploratory_analysis.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Customer Churn - Exploratory Data Analysis

This notebook explores the **synthetic** customer churn dataset used by the project
(`data/customer_churn.csv`). Every customer is fictional - there is no real personal data.

**Goals**
1. Understand the raw data and its quality problems.
2. Clean it with the same code the model uses (`src/data_processing.py`).
3. Explore how churn relates to contract, tenure, charges, payment, services and segments.
4. Summarise observations that motivate the feature engineering and modelling.

> ⚠️ **Correlation is not causation.** The patterns below show *associations* in the data.
> A group with a higher churn rate does not prove that the attribute *causes* churn."""),
    code("""import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from src.config import DATA_PATH, NUMERIC_COLUMNS
from src.data_processing import load_raw_data, clean_data
from src.feature_engineering import engineer_features
from src.insights import churn_rate_by, highest_churn_segment
from src import plots

pd.set_option("display.max_columns", 30)
%matplotlib inline"""),
    md("## 1. Raw data and data-quality check"),
    code("""raw = load_raw_data(DATA_PATH)
print(f"{len(raw):,} rows x {raw.shape[1]} columns")
raw.head()"""),
    code("""quality = pd.DataFrame({
    "missing": raw.isna().sum(),
    "blank_strings": (raw.apply(lambda s: s.astype(str).str.strip() == "")).sum(),
    "unique_values": raw.nunique(),
})
print("Exact duplicate rows:", raw.duplicated().sum())
quality[(quality.missing > 0) | (quality.blank_strings > 0)]"""),
    code("""# Inconsistent category spellings in the raw data
for col in ["contract_type", "internet_service", "gender", "partner"]:
    print(col, "->", sorted(raw[col].dropna().unique()))"""),
    md("""**Observations**
- A few percent of values are missing in several columns, and some `total_charges` are blank strings.
- The same category is written in several ways (e.g. `MTM`, `month-to-month`, `Month to Month`).
- There are exact duplicate rows and a handful of impossible numbers (e.g. negative charges, age 999).

## 2. Cleaning

The cleaning rules are documented in `src/data_processing.py` and the README. Invalid values are
set to *missing* rather than deleting whole customers; the model pipeline later imputes them using
statistics learned from the **training split only** (no data leakage)."""),
    code("""df, report = clean_data(raw)
pd.DataFrame(report.as_rows(), columns=["Step", "Count"])"""),
    code("""print("Remaining missing values (imputed later inside the pipeline):")
pd.Series(report.missing_after_cleaning, name="missing").to_frame()"""),
    md("## 3. Churn distribution"),
    code("""print(f"Churn rate: {df.churn.mean():.1%}")
plots.churn_distribution(df)"""),
    md("""About one in four customers churned. The classes are **imbalanced**, so accuracy alone would be
misleading: predicting "stays" for everybody would already be ~75% accurate while catching no churners.
Model evaluation therefore focuses on precision, recall, F1 and ROC-AUC."""),
    md("## 4. Churn by contract type"),
    code("""rates = churn_rate_by(df, "contract_type")
display(rates)
plots.churn_rate_bar(rates, "contract_type", "Churn rate by contract type", overall=df.churn.mean())"""),
    md("## 5. Churn by tenure"),
    code("""eng = engineer_features(df)
display(churn_rate_by(eng, "tenure_group"))
plots.histogram_by_churn(df, "tenure_months", title="Tenure by churn status", xlabel="Tenure (months)")"""),
    md("## 6. Churn by monthly charges"),
    code("""eng["charge_band"] = pd.cut(eng.monthly_charges, [0, 35, 60, 85, 200],
                           labels=["< $35", "$35-60", "$60-85", "$85+"])
display(churn_rate_by(eng, "charge_band"))
plots.histogram_by_churn(df, "monthly_charges", title="Monthly charges by churn status",
                         xlabel="Monthly charges ($)")"""),
    md("## 7. Churn by payment method"),
    code("""rates = churn_rate_by(df, "payment_method")
plots.churn_rate_bar(rates, "payment_method", "Churn rate by payment method", overall=df.churn.mean())"""),
    md("## 8. Churn by service type"),
    code("""for col in ["internet_service", "tech_support", "online_security"]:
    display(churn_rate_by(df, col))
plots.churn_rate_bar(churn_rate_by(df, "internet_service"), "internet_service",
                     "Churn rate by internet service", overall=df.churn.mean())"""),
    code("""display(churn_rate_by(eng, "service_count"))"""),
    md("## 9. Churn by customer segment"),
    code("""for col in ["age_group", "region", "partner", "dependents"]:
    display(churn_rate_by(eng, col))
seg = highest_churn_segment(df)
print("Highest-churn two-attribute segment:", seg)"""),
    md("## 10. Support and satisfaction"),
    code("""display(churn_rate_by(df, "satisfaction_score"))
eng["tickets_band"] = pd.cut(eng.support_tickets, [-1, 0, 2, 4, 50], labels=["0", "1-2", "3-4", "5+"])
display(churn_rate_by(eng, "tickets_band"))"""),
    md("## 11. Signup date"),
    code("""plots.signup_trend(df)"""),
    md("""Customers who signed up recently churn more - but recent sign-ups also have short tenure, so this
largely repeats the tenure pattern. `signup_date` is therefore kept for analysis but **not** used as a
model feature (it would duplicate tenure and ties the model to a specific snapshot date)."""),
    md("## 12. Correlations (numeric features)"),
    code("""cols = NUMERIC_COLUMNS + ["service_count", "avg_monthly_spend", "churn"]
display(eng[cols].corr()["churn"].drop("churn").sort_values())
plots.correlation_heatmap(eng, cols)"""),
    md("""`monthly_charges`, `service_count` and `avg_monthly_spend` are strongly correlated with each other
(more services -> higher bill), and `total_charges` is driven by tenure. Correlated inputs make individual
Logistic Regression coefficients harder to interpret, which is one reason the project also uses
SHAP and permutation importance.

## 13. Key observations (associations, not causes)

- **Contract type** shows the largest difference: month-to-month customers churn far more than
  customers on one- or two-year contracts.
- **Tenure**: churn is concentrated in the first year and falls steadily with tenure.
- **Monthly charges**: churners tend to have higher monthly bills.
- **Payment method**: electronic-check customers churn more than automatic-payment customers.
- **Services**: fiber-optic customers churn more; customers with tech support / online security churn less.
- **Support experience**: more support tickets and lower satisfaction scores go with higher churn.
- Demographics (gender, region) show small differences.

These observations motivated the engineered features (tenure groups, service count, average spend,
new-customer flag, protection bundle) and the business recommendations in the dashboard. Whether any
of these factors *causes* churn would need a controlled experiment (e.g. an A/B test of a retention offer)."""),
]

# A trailing ";" stops Jupyter from displaying each returned figure twice.
for cell in cells:
    if cell.cell_type == "code":
        last = cell.source.rstrip().split("\n")
        stmt_start = max(i for i, line in enumerate(last) if not line.startswith(" "))
        if last[stmt_start].startswith("plots."):
            cell.source = cell.source.rstrip() + ";"

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

if __name__ == "__main__":
    if "--no-execute" not in sys.argv:
        from nbconvert.preprocessors import ExecutePreprocessor

        ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(HERE)}}
        )
    nbf.write(nb, OUT)
    print(f"Wrote {OUT}")
