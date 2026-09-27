"""Train, tune, evaluate and save the churn models.

Usage (from the project root):
    python src/train_models.py

Workflow
--------
1. Load and clean the raw data (see src/data_processing.py).
2. Stratified train/test split (80/20, random_state=42). The test set is
   locked away until final evaluation.
3. Build one Pipeline per model: feature engineering -> imputation /
   scaling / one-hot encoding -> classifier. Everything that learns from
   data is fitted inside the pipeline on training folds only (no leakage).
4. 5-fold stratified cross-validation of the four baseline models on the
   training set.
5. Randomized hyperparameter search for the three strongest baseline models
   (by CV ROC-AUC). A tuned model replaces its baseline only if its CV score
   is better.
6. Choose each model's decision threshold (max F1 on out-of-fold training
   predictions), fit on the full training set and evaluate on the test set.
7. Select the primary model by CV ROC-AUC (see evaluate_models.py).
8. Compute permutation importance and SHAP importance for the primary model.
9. Save pipelines (joblib), metrics (JSON) and report figures.
"""

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src import plots  # noqa: E402
from src.config import (  # noqa: E402
    ALL_MODELS_PATH,
    CLASSIFICATION_THRESHOLD,
    CV_FOLDS,
    DATA_PATH,
    EVALUATION_PATH,
    FIGURES_DIR,
    HIGH_RISK_MIN,
    ID_COLUMN,
    LOW_RISK_MAX,
    MODELS_DIR,
    PRIMARY_MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_processing import load_clean_data  # noqa: E402
from src.evaluate_models import (  # noqa: E402
    comparison_table,
    evaluate_predictions,
    select_primary_model,
)
from src.explainability import SHAP_AVAILABLE, global_shap_importance, model_feature_importance  # noqa: E402
from src.feature_engineering import build_pipeline  # noqa: E402
from src.insights import churn_rate_by  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

SCORING = {"roc_auc": "roc_auc", "f1": "f1", "precision": "precision", "recall": "recall"}
N_ITER_SEARCH = 20


def make_models():
    """Baseline models with sensible defaults.

    No class weights or resampling are used on purpose: re-weighting the
    classes inflates the predicted probabilities (a 25% base rate would look
    like ~50%), which would make the "churn probability" shown to users
    misleading. Class imbalance is instead handled by (a) a stratified split,
    (b) imbalance-aware metrics (precision, recall, F1, ROC-AUC) and (c) a
    decision threshold chosen by cross-validation to maximise F1.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1, tree_method="hist",
        ),
    }


PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {"model__C": loguniform(1e-3, 1e2)},
    "Decision Tree": {
        "model__max_depth": randint(3, 15),
        "model__min_samples_leaf": randint(5, 80),
        "model__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators": randint(150, 350),
        "model__max_depth": [None, 6, 10, 14, 20],
        "model__min_samples_leaf": randint(1, 20),
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    "XGBoost": {
        "model__n_estimators": randint(150, 600),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": randint(1, 10),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.1, 10),
    },
}


def choose_threshold(y_true, proba, grid=np.arange(0.10, 0.71, 0.01)):
    """Pick the probability threshold that maximises F1 on out-of-fold
    TRAINING predictions (the test set is never used for this)."""
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return round(float(grid[int(np.argmax(scores))]), 2)


def cv_summary(cv_result):
    return {
        f"{m}_{stat}": float(fn(cv_result[f"test_{m}"]))
        for m in SCORING
        for stat, fn in (("mean", np.mean), ("std", np.std))
    }


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean ------------------------------------------------------
    log(f"Loading data from {DATA_PATH}")
    df, report = load_clean_data(DATA_PATH)
    log(f"Cleaned data: {report.rows_in:,} rows in -> {report.rows_out:,} rows out")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 2. Stratified split ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log(f"Train: {len(X_train):,} rows ({y_train.mean():.1%} churn) | "
        f"Test: {len(X_test):,} rows ({y_test.mean():.1%} churn)")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 3-4. Baseline pipelines + cross-validation ---------------------------
    pipelines, results = {}, {}
    for name, model in make_models().items():
        pipe = build_pipeline(model)
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        pipelines[name] = pipe
        results[name] = {"cv": cv_summary(cv_res), "baseline_cv": cv_summary(cv_res),
                         "tuned": False, "best_params": {}}
        log(f"Baseline {name:<20} CV ROC-AUC {results[name]['cv']['roc_auc_mean']:.4f} "
            f"± {results[name]['cv']['roc_auc_std']:.4f}")

    # 5. Tune the three strongest baselines ----------------------------------
    to_tune = sorted(results, key=lambda n: results[n]["cv"]["roc_auc_mean"], reverse=True)[:3]
    for name in to_tune:
        log(f"Tuning {name} ({N_ITER_SEARCH} random candidates x {CV_FOLDS} folds)...")
        search = RandomizedSearchCV(
            pipelines[name], PARAM_DISTRIBUTIONS[name], n_iter=N_ITER_SEARCH, cv=cv,
            scoring=SCORING, refit="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        i = search.best_index_
        tuned_cv = {
            f"{m}_{stat}": float(search.cv_results_[f"{stat}_test_{m}"][i])
            for m in SCORING for stat in ("mean", "std")
        }
        improved = tuned_cv["roc_auc_mean"] > results[name]["cv"]["roc_auc_mean"]
        log(f"  best CV ROC-AUC {tuned_cv['roc_auc_mean']:.4f} "
            f"({'kept tuned' if improved else 'baseline was better - kept baseline'})")
        if improved:
            pipelines[name] = search.best_estimator_
            results[name].update(
                cv=tuned_cv, tuned=True,
                best_params={k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                             for k, v in search.best_params_.items()},
            )

    # 6. Decision threshold (from out-of-fold training predictions), final
    #    fit on the full training set and test evaluation --------------------
    for name, pipe in pipelines.items():
        oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        threshold = choose_threshold(y_train, oof)
        results[name]["threshold"] = threshold
        if not results[name]["tuned"]:
            pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name]["test"] = evaluate_predictions(y_test, proba, threshold)
        results[name]["test_at_0_5"] = {
            k: v for k, v in evaluate_predictions(y_test, proba, 0.5).items()
            if k in ("Accuracy", "Precision", "Recall", "F1")
        }
        train_auc = evaluate_predictions(y_train, pipe.predict_proba(X_train)[:, 1])["ROC-AUC"]
        results[name]["train_roc_auc"] = train_auc
        t = results[name]["test"]
        log(f"Test  {name:<20} Acc {t['Accuracy']:.3f}  Prec {t['Precision']:.3f}  "
            f"Rec {t['Recall']:.3f}  F1 {t['F1']:.3f}  AUC {t['ROC-AUC']:.3f} "
            f"thr {results[name]['threshold']:.2f} "
            f"(train AUC {train_auc:.3f})")

    # 7. Primary model selection --------------------------------------------
    primary_name, reason = select_primary_model(results, float(y_train.mean()))
    primary = pipelines[primary_name]
    log(f"Primary model: {primary_name}. {reason}")

    # 8. Explainability for the primary model -------------------------------
    native_importance = {}
    for name, pipe in pipelines.items():
        imp, method = model_feature_importance(pipe)
        native_importance[name] = {"method": method, "values": imp.to_dict(orient="records")}

    log("Computing permutation importance on the test set...")
    perm = permutation_importance(
        primary, X_test, y_test, scoring="roc_auc", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    feature_cols = [c for c in X_test.columns if c not in (ID_COLUMN, "signup_date")]
    perm_df = (
        pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean,
                      "std": perm.importances_std})
        .loc[lambda d: d["feature"].isin(feature_cols)]
        .sort_values("importance", ascending=False)
    )

    shap_records = []
    if SHAP_AVAILABLE:
        log("Computing SHAP values on a test sample...")
        sample = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
        shap_imp = global_shap_importance(primary, sample, background=X_train)
        shap_records = shap_imp.to_dict(orient="records")

    # 9. Save artefacts --------------------------------------------------------
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle = {
        "pipeline": primary,
        "model_name": primary_name,
        "selection_reason": reason,
        "classification_threshold": results[primary_name]["threshold"],
        "risk_thresholds": {"low_max": LOW_RISK_MAX, "high_min": HIGH_RISK_MIN},
        "reference_values": {
            # Used by the recommendation rules ("top 25% monthly charges").
            "high_monthly_charges": float(X_train["monthly_charges"].quantile(0.75)),
        },
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, PRIMARY_MODEL_PATH, compress=3)
    joblib.dump(pipelines, ALL_MODELS_PATH, compress=3)

    evaluation = {
        "trained_at": trained_at,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "classification_threshold": results[primary_name]["threshold"],
        "default_threshold": CLASSIFICATION_THRESHOLD,
        "primary_model": primary_name,
        "selection_reason": reason,
        "tuned_models": to_tune,
        "models": results,
        "native_importance": native_importance,
        "permutation_importance": perm_df.to_dict(orient="records"),
        "shap_importance": shap_records,
        "test_customer_ids": X_test[ID_COLUMN].tolist(),
        "cleaning_report": report.as_rows(),
    }
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2, default=str))

    # Human-readable reports committed to the repo for GitHub readers.
    table = comparison_table(results).round(4)
    table.to_csv(REPORTS_DIR / "model_comparison.csv")
    plots.save(plots.model_comparison_bars(table), FIGURES_DIR / "model_comparison.png")
    plots.save(plots.roc_curves(results), FIGURES_DIR / "roc_curves.png")
    plots.save(plots.confusion_matrix_plot(results[primary_name]["test"]["confusion_matrix"],
                                           f"Confusion matrix - {primary_name}"),
               FIGURES_DIR / "confusion_matrix_primary.png")
    plots.save(plots.churn_rate_bar(churn_rate_by(df, "contract_type"), "contract_type",
                                    "Churn rate by contract type", overall=y.mean()),
               FIGURES_DIR / "churn_by_contract.png")
    plots.save(plots.churn_rate_bar(churn_rate_by(df, "tenure_group"), "tenure_group",
                                    "Churn rate by tenure group", overall=y.mean()),
               FIGURES_DIR / "churn_by_tenure.png")
    if shap_records:
        plots.save(plots.importance_bar(pd.DataFrame(shap_records), "mean_abs_shap",
                                        f"SHAP importance - {primary_name}",
                                        xlabel="Mean |SHAP value|"),
                   FIGURES_DIR / "shap_importance.png")

    log("\nModel comparison (test set):\n" + table.to_string())
    log(f"Saved: {PRIMARY_MODEL_PATH.name}, {ALL_MODELS_PATH.name}, {EVALUATION_PATH.name}")
    log(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
