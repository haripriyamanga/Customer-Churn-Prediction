"""Central configuration: paths, schema, random seed and risk thresholds.

Keeping these values in one place means the training script, the dashboard,
the notebook and the tests all agree on the same definitions.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_PATH = DATA_DIR / "customer_churn.csv"
PRIMARY_MODEL_PATH = MODELS_DIR / "churn_model.joblib"
ALL_MODELS_PATH = MODELS_DIR / "all_models.joblib"
EVALUATION_PATH = MODELS_DIR / "evaluation.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Schema of the raw dataset
# ---------------------------------------------------------------------------
ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"
DATE_COLUMNS = ["signup_date"]

NUMERIC_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "satisfaction_score",
    "late_payments",
]

# Canonical category values. Cleaning maps messy spellings onto these.
CATEGORICAL_VALUES = {
    "gender": ["Female", "Male"],
    "partner": ["No", "Yes"],
    "dependents": ["No", "Yes"],
    "region": ["East", "North", "South", "West"],
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "payment_method": ["Bank transfer", "Credit card", "Electronic check", "Mailed check"],
    "paperless_billing": ["No", "Yes"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "phone_service": ["No", "Yes"],
    "online_security": ["No", "Yes"],
    "online_backup": ["No", "Yes"],
    "device_protection": ["No", "Yes"],
    "tech_support": ["No", "Yes"],
    "streaming_tv": ["No", "Yes"],
    "streaming_movies": ["No", "Yes"],
}
CATEGORICAL_COLUMNS = list(CATEGORICAL_VALUES)

# Service flags that are counted to build the "service_count" feature.
SERVICE_COLUMNS = [
    "phone_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Valid ranges for numeric columns. Values outside the range are treated as
# data-entry errors and set to missing (they are then imputed inside the
# model pipeline, using statistics learned from the training data only).
NUMERIC_RANGES = {
    "age": (18, 100),
    "tenure_months": (0, 120),
    "monthly_charges": (0, 500),
    "total_charges": (0, 60_000),
    "support_tickets": (0, 50),
    "satisfaction_score": (1, 5),
    "late_payments": (0, 24),
}

# Columns a user must supply to get a prediction (CSV upload / form).
REQUIRED_INPUT_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# ---------------------------------------------------------------------------
# Risk categories (applied to the predicted churn probability)
# ---------------------------------------------------------------------------
# probability <  LOW_RISK_MAX                  -> Low Risk
# LOW_RISK_MAX <= probability < HIGH_RISK_MIN  -> Medium Risk
# probability >= HIGH_RISK_MIN                 -> High Risk
#
# 0.25 is roughly the average churn rate, so "Low Risk" means "below the
# typical customer". 0.60 means the model thinks churn is more likely than not
# by a clear margin. These are business choices and can be changed here.
LOW_RISK_MAX = 0.25
HIGH_RISK_MIN = 0.60

# Default probability at or above which the label is "Likely to churn".
# Training replaces this per model with the threshold that maximised F1 in
# cross-validation (stored with the saved model).
CLASSIFICATION_THRESHOLD = 0.50
