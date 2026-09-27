"""Model evaluation: metrics, confusion matrices, ROC curves and selection.

Every number is computed from model predictions - nothing is hard-coded.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CLASSIFICATION_THRESHOLD

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def evaluate_predictions(y_true, y_proba, threshold=CLASSIFICATION_THRESHOLD):
    """Compute all evaluation metrics for one model.

    Returns a dict with the headline metrics, the confusion matrix
    (as TN, FP, FN, TP counts) and the ROC curve points.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    fpr, tpr, _ = roc_curve(y_true, y_proba) if has_both_classes else ([], [], [])

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)) if has_both_classes else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # Downsample the ROC curve so the saved JSON stays small.
        "roc_curve": _downsample_curve(np.asarray(fpr), np.asarray(tpr)),
        "threshold": float(threshold),
    }


def _downsample_curve(fpr, tpr, max_points=200):
    if len(fpr) <= max_points:
        idx = np.arange(len(fpr))
    else:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
    return {"fpr": [round(float(v), 5) for v in fpr[idx]],
            "tpr": [round(float(v), 5) for v in tpr[idx]]}


def comparison_table(results):
    """Build the Model | Accuracy | Precision | Recall | F1 | ROC-AUC table.

    ``results`` maps model name -> evaluation dict (from
    :func:`evaluate_predictions`, optionally with ``cv_roc_auc_mean``).
    """
    rows = []
    for name, res in results.items():
        row = {"Model": name, **{m: res["test"][m] for m in METRIC_COLUMNS}}
        if "threshold" in res:
            row["Threshold"] = res["threshold"]
        if "cv" in res:
            row["CV ROC-AUC (mean)"] = res["cv"]["roc_auc_mean"]
            row["CV ROC-AUC (std)"] = res["cv"]["roc_auc_std"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def select_primary_model(results, churn_rate=None):
    """Choose the primary model.

    Rule: highest mean cross-validated ROC-AUC on the TRAINING data. ROC-AUC
    measures how well a model ranks churners above non-churners across all
    thresholds and is not fooled by class imbalance the way accuracy is.
    Selecting on cross-validation (not the test set) keeps the test set as an
    unbiased final check. If two models are within 0.005 CV ROC-AUC, the one
    with the higher cross-validated F1 wins (better precision/recall balance).

    Returns (model_name, explanation_text).
    """
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["cv"]["roc_auc_mean"], kv[1]["cv"]["f1_mean"]),
        reverse=True,
    )
    best_name, best = ranked[0]
    reason = (
        f"{best_name} has the highest mean 5-fold cross-validated ROC-AUC on the "
        f"training data ({best['cv']['roc_auc_mean']:.3f})."
    )
    if len(ranked) > 1:
        runner_name, runner = ranked[1]
        gap = best["cv"]["roc_auc_mean"] - runner["cv"]["roc_auc_mean"]
        if gap < 0.005 and runner["cv"]["f1_mean"] > best["cv"]["f1_mean"]:
            best_name, best = runner_name, runner
            reason = (
                f"{runner_name} and {ranked[0][0]} are within 0.005 CV ROC-AUC, so the "
                f"tie-break (higher CV F1: {runner['cv']['f1_mean']:.3f}) selects {runner_name}."
            )
        else:
            reason += f" The runner-up is {runner_name} ({runner['cv']['roc_auc_mean']:.3f})."
    if churn_rate is not None:
        reason += (
            f" Accuracy was not used to choose, because with {churn_rate:.0%} churners a model "
            f"that predicts 'no churn' for everyone would already be {1 - churn_rate:.0%} accurate."
        )
    return best_name, reason


def confusion_matrix_frame(cm):
    """Return a labelled 2x2 DataFrame from a confusion-matrix dict."""
    return pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual: Stayed", "Actual: Churned"],
        columns=["Predicted: Stay", "Predicted: Churn"],
    )
