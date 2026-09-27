"""Data loading, cleaning and validation.

Cleaning decisions (also documented in the README):

1. Whitespace is stripped and empty strings are treated as missing.
2. Category spellings are standardised (e.g. "MTM", "month-to-month"
   -> "Month-to-month"; "F" -> "Female"). Values that cannot be mapped to a
   known category are set to missing rather than guessed.
3. Numeric columns are coerced to numbers. Text that is not a number becomes
   missing. Values outside a plausible range (e.g. age 999, negative
   charges) are treated as data-entry errors and set to missing.
4. `total_charges` blanks for customers with 0 months of tenure are set to
   0 - they have not been billed yet. This is a deterministic business rule,
   not a statistic learned from the data, so it cannot leak information.
5. Remaining missing values are NOT filled here. They are imputed inside the
   model pipeline, which learns the fill values from the training split only
   (this prevents data leakage).
6. Exact duplicate rows are removed (they are export artefacts). If the same
   customer ID still appears more than once, the first record is kept.
7. Rows with a missing/unknown churn label are removed from the modelling
   data, because a model cannot learn from an unknown outcome. They are
   counted in the cleaning report, never dropped silently.
8. `signup_date` is parsed to a date. Unparseable dates become missing. The
   date is kept for EDA but not used as a model feature (tenure already
   captures how long a customer has been with the company).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_VALUES,
    DATA_PATH,
    DATE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    NUMERIC_RANGES,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMN,
)

# Known messy spellings -> canonical value (matched after lower-casing).
CATEGORY_ALIASES = {
    "gender": {"f": "Female", "m": "Male"},
    "contract_type": {
        "month to month": "Month-to-month",
        "month-to-month": "Month-to-month",
        "mtm": "Month-to-month",
        "monthly": "Month-to-month",
        "1 year": "One year",
        "one-year": "One year",
        "2 year": "Two year",
        "two-year": "Two year",
    },
    "internet_service": {
        "fibre": "Fiber optic",
        "fiber": "Fiber optic",
        "fibre optic": "Fiber optic",
        "none": "No",
        "no internet service": "No",
    },
    "payment_method": {
        "bank transfer (automatic)": "Bank transfer",
        "credit card (automatic)": "Credit card",
    },
}
YES_NO_ALIASES = {
    "y": "Yes",
    "yes": "Yes",
    "true": "Yes",
    "1": "Yes",
    "n": "No",
    "no": "No",
    "false": "No",
    "0": "No",
    "no internet service": "No",
    "no phone service": "No",
}
TARGET_ALIASES = {"yes": 1, "y": 1, "1": 1, "true": 1, "churned": 1,
                  "no": 0, "n": 0, "0": 0, "false": 0, "retained": 0}


@dataclass
class CleaningReport:
    """Counts of every change made during cleaning, shown in the dashboard."""

    rows_in: int = 0
    rows_out: int = 0
    duplicate_rows_removed: int = 0
    duplicate_ids_removed: int = 0
    missing_target_removed: int = 0
    categories_standardised: dict = field(default_factory=dict)
    unknown_categories_set_missing: dict = field(default_factory=dict)
    invalid_numbers_set_missing: dict = field(default_factory=dict)
    total_charges_zero_filled: int = 0
    invalid_dates: int = 0
    missing_after_cleaning: dict = field(default_factory=dict)

    def as_rows(self):
        """Return the report as (step, detail) rows for display."""
        rows = [
            ("Rows loaded", f"{self.rows_in:,}"),
            ("Exact duplicate rows removed", f"{self.duplicate_rows_removed:,}"),
            ("Repeated customer IDs removed (first kept)", f"{self.duplicate_ids_removed:,}"),
            ("Rows with unknown churn label removed", f"{self.missing_target_removed:,}"),
            ("Blank total_charges set to 0 (tenure = 0)", f"{self.total_charges_zero_filled:,}"),
            ("Unparseable signup dates set to missing", f"{self.invalid_dates:,}"),
        ]
        for col, n in self.categories_standardised.items():
            rows.append((f"Category spellings standardised: {col}", f"{n:,}"))
        for col, n in self.unknown_categories_set_missing.items():
            rows.append((f"Unknown categories set to missing: {col}", f"{n:,}"))
        for col, n in self.invalid_numbers_set_missing.items():
            rows.append((f"Invalid / out-of-range values set to missing: {col}", f"{n:,}"))
        rows.append(("Rows after cleaning", f"{self.rows_out:,}"))
        return rows


def load_raw_data(path=DATA_PATH):
    """Load the raw CSV exactly as stored (all columns read as text first)."""
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def _standardise_category(series, col):
    """Map messy spellings onto canonical category values."""
    valid = CATEGORICAL_VALUES[col]
    lookup = {v.lower(): v for v in valid}
    aliases = YES_NO_ALIASES if set(valid) == {"No", "Yes"} else CATEGORY_ALIASES.get(col, {})

    def fix(value):
        if pd.isna(value):
            return np.nan
        key = str(value).strip().lower()
        if key in lookup:
            return lookup[key]
        if key in aliases:
            return aliases[key]
        return "__UNKNOWN__"

    return series.map(fix)


def clean_categoricals(df, report=None):
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        before = df[col].copy()
        fixed = _standardise_category(before, col)
        unknown = fixed == "__UNKNOWN__"
        fixed = fixed.where(~unknown, np.nan)
        changed = before.notna() & fixed.notna() & (before.astype(str) != fixed.astype(str))
        if report is not None:
            if changed.sum():
                report.categories_standardised[col] = int(changed.sum())
            if unknown.sum():
                report.unknown_categories_set_missing[col] = int(unknown.sum())
        df[col] = fixed
    return df


def clean_numerics(df, report=None):
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        raw = df[col]
        values = pd.to_numeric(raw, errors="coerce")
        # Text that was present but not numeric (excluding blanks handled later).
        non_numeric = raw.notna() & values.isna()
        low, high = NUMERIC_RANGES[col]
        out_of_range = values.notna() & ((values < low) | (values > high))
        values = values.mask(out_of_range)
        if col == "total_charges" and "tenure_months" in df.columns:
            tenure = pd.to_numeric(df["tenure_months"], errors="coerce")
            zero_fill = values.isna() & (tenure == 0)
            values = values.mask(zero_fill, 0.0)
            non_numeric = non_numeric & ~zero_fill
            if report is not None:
                report.total_charges_zero_filled = int(zero_fill.sum())
        n_invalid = int(non_numeric.sum() + out_of_range.sum())
        if report is not None and n_invalid:
            report.invalid_numbers_set_missing[col] = n_invalid
        df[col] = values.astype(float)
    return df


def clean_dates(df, report=None):
    df = df.copy()
    for col in DATE_COLUMNS:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
        if report is not None:
            report.invalid_dates += int((df[col].notna() & parsed.isna()).sum())
        df[col] = parsed
    return df


def encode_target(series):
    """Convert churn labels such as Yes/No to 1/0 (unknown -> NaN)."""
    return series.map(lambda v: TARGET_ALIASES.get(str(v).strip().lower(), np.nan)
                      if pd.notna(v) else np.nan)


def clean_data(raw, require_target=True):
    """Run the full cleaning procedure.

    Parameters
    ----------
    raw : DataFrame
        Raw data (as loaded by :func:`load_raw_data` or uploaded by a user).
    require_target : bool
        If True, rows without a valid churn label are removed (needed for
        training). If False (scoring new customers) the label is optional.

    Returns
    -------
    (DataFrame, CleaningReport)
    """
    report = CleaningReport(rows_in=len(raw))
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Strip whitespace in text columns; empty strings become missing.
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("object").map(
                lambda v: v.strip() if isinstance(v, str) else v
            ).replace("", np.nan)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report.duplicate_rows_removed = before - len(df)

    if ID_COLUMN in df.columns:
        before = len(df)
        df = df[~(df[ID_COLUMN].notna() & df[ID_COLUMN].duplicated(keep="first"))]
        df = df.reset_index(drop=True)
        report.duplicate_ids_removed = before - len(df)

    df = clean_categoricals(df, report)
    df = clean_numerics(df, report)
    df = clean_dates(df, report)

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = encode_target(df[TARGET_COLUMN])
        if require_target:
            missing = df[TARGET_COLUMN].isna()
            report.missing_target_removed = int(missing.sum())
            df = df[~missing].reset_index(drop=True)
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    elif require_target:
        raise ValueError(f"The data has no '{TARGET_COLUMN}' column, which is needed for training.")

    report.rows_out = len(df)
    missing = df.isna().sum()
    report.missing_after_cleaning = {k: int(v) for k, v in missing[missing > 0].items()}
    return df, report


def load_clean_data(path=DATA_PATH):
    """Convenience helper: load the raw CSV and clean it."""
    return clean_data(load_raw_data(path))


# ---------------------------------------------------------------------------
# Validation of user-supplied data (CSV upload / prediction form)
# ---------------------------------------------------------------------------
def validate_input_data(raw, max_missing_fraction=0.5):
    """Check that uploaded data matches the expected structure.

    Returns a dict with ``errors`` (blocking problems) and ``warnings``
    (things the user should know, but which the pipeline can handle).
    """
    errors, warnings = [], []
    if raw is None or len(raw) == 0:
        return {"errors": ["The file contains no rows."], "warnings": []}

    columns = [str(c).strip() for c in raw.columns]
    missing_cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in columns]
    if missing_cols:
        errors.append("Missing required columns: " + ", ".join(missing_cols))
        return {"errors": errors, "warnings": warnings}

    df = raw.copy()
    df.columns = columns

    for col in NUMERIC_COLUMNS:
        text = df[col].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
        text = text.replace("", np.nan)
        values = pd.to_numeric(text, errors="coerce")
        bad = int((text.notna() & values.isna()).sum())
        if bad and bad == int(text.notna().sum()):
            errors.append(f"Column '{col}' must be numeric, but none of its values are numbers.")
        elif bad:
            warnings.append(f"Column '{col}': {bad} non-numeric value(s) will be treated as missing.")
        low, high = NUMERIC_RANGES[col]
        out = int((values.notna() & ((values < low) | (values > high))).sum())
        if out:
            warnings.append(
                f"Column '{col}': {out} value(s) outside the valid range {low}-{high} "
                "will be treated as missing."
            )

    for col in CATEGORICAL_COLUMNS:
        std = _standardise_category(df[col].replace("", np.nan), col)
        unknown = sorted({str(v) for v, s in zip(df[col], std) if s == "__UNKNOWN__"})
        if unknown and len(unknown) == df[col].dropna().nunique():
            errors.append(
                f"Column '{col}' has no recognised values. Expected one of: "
                + ", ".join(CATEGORICAL_VALUES[col])
            )
        elif unknown:
            shown = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
            warnings.append(
                f"Column '{col}': unrecognised value(s) {shown} will be treated as missing. "
                f"Expected one of: {', '.join(CATEGORICAL_VALUES[col])}."
            )

    for col in REQUIRED_INPUT_COLUMNS:
        frac = df[col].replace("", np.nan).isna().mean()
        if frac > max_missing_fraction:
            errors.append(f"Column '{col}' is {frac:.0%} empty - too many missing values to score.")
        elif frac > 0:
            warnings.append(
                f"Column '{col}' has {frac:.1%} missing values; they will be imputed "
                "using values learned from the training data."
            )

    return {"errors": errors, "warnings": warnings}
