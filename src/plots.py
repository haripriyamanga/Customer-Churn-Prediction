"""Matplotlib charts shared by the dashboard, the notebook and the training report."""

import matplotlib

matplotlib.use("Agg")  # headless backend: works in Streamlit, tests and scripts

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Colour roles (colour-blind-safe categorical order; status colours for risk).
STAYED_COLOR = "#2a78d6"  # blue
CHURN_COLOR = "#eb6834"   # orange
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RISK_COLORS = {"Low Risk": "#0ca30c", "Medium Risk": "#fab219", "High Risk": "#d03b3b"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
INCREASE_COLOR = "#d03b3b"
DECREASE_COLOR = "#2a78d6"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=TEXT, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    return ax


def _fig(width=7, height=4):
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    return fig, ax


def churn_distribution(df, target="churn"):
    counts = df[target].map({0: "Stayed", 1: "Churned"}).value_counts().reindex(["Stayed", "Churned"])
    fig, ax = _fig(5, 3.6)
    bars = ax.bar(counts.index, counts.values, color=[STAYED_COLOR, CHURN_COLOR], width=0.55)
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.annotate(f"{value:,}\n({value / total:.1%})", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9, color=TEXT, xytext=(0, 3),
                    textcoords="offset points")
    ax.set_ylim(0, counts.max() * 1.22)
    _style(ax, "Churn distribution", None, "Customers")
    fig.tight_layout()
    return fig


def churn_rate_bar(rates, column, title=None, overall=None):
    """Horizontal bar chart of churn rate per category (from insights.churn_rate_by)."""
    data = rates.sort_values("churn_rate")
    fig, ax = _fig(7, 0.55 * len(data) + 1.4)
    ax.barh(data[column].astype(str), data["churn_rate"] * 100, color=CHURN_COLOR, height=0.6)
    for y, (rate, n) in enumerate(zip(data["churn_rate"], data["customers"])):
        ax.annotate(f"{rate:.1%}  (n={n:,})", (rate * 100, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)
    if overall is not None:
        ax.axvline(overall * 100, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"overall {overall:.1%}", (overall * 100, len(data) - 0.5), fontsize=8,
                    color=MUTED, xytext=(3, 0), textcoords="offset points")
    ax.set_xlim(0, max(data["churn_rate"].max() * 100 * 1.35, 5))
    _style(ax, title or f"Churn rate by {column}", "Churn rate (%)", None)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def histogram_by_churn(df, column, bins=30, title=None, xlabel=None, target="churn"):
    fig, ax = _fig(7, 3.8)
    values = df[[column, target]].dropna()
    edges = np.histogram_bin_edges(values[column], bins=bins)
    for label, color, name in [(0, STAYED_COLOR, "Stayed"), (1, CHURN_COLOR, "Churned")]:
        ax.hist(values.loc[values[target] == label, column], bins=edges, alpha=0.6,
                color=color, label=name, density=True, edgecolor="white", linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, title or f"{column} by churn status", xlabel or column, "Density")
    fig.tight_layout()
    return fig


def correlation_heatmap(df, columns):
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=110)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(columns)), columns, fontsize=8, color=MUTED)
    for i in range(len(columns)):
        for j in range(len(columns)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix (numeric features)", loc="left", fontsize=12,
                 color=TEXT, fontweight="bold")
    fig.tight_layout()
    return fig


def model_comparison_bars(table, metrics=("Precision", "Recall", "F1", "ROC-AUC")):
    fig, ax = _fig(8, 4)
    metrics = list(metrics)
    n_models = len(table)
    width = 0.8 / n_models
    x = np.arange(len(metrics))
    for i, (name, row) in enumerate(table.iterrows()):
        ax.bar(x + i * width - 0.4 + width / 2, [row[m] for m in metrics], width * 0.92,
               label=name, color=MODEL_COLORS[i % len(MODEL_COLORS)])
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, ncol=n_models, loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    _style(ax, "Model comparison on the test set", None, "Score")
    fig.tight_layout()
    return fig


def roc_curves(results):
    fig, ax = _fig(6, 5)
    for i, (name, res) in enumerate(results.items()):
        curve = res["test"]["roc_curve"]
        ax.plot(curve["fpr"], curve["tpr"], linewidth=2, color=MODEL_COLORS[i % 4],
                label=f"{name} (AUC = {res['test']['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Random guess")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, "ROC curves (test set)", "False positive rate", "True positive rate (recall)")
    fig.tight_layout()
    return fig


def confusion_matrix_plot(cm, title):
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    labels = np.array([["True negative", "False positive"], ["False negative", "True positive"]])
    fig, ax = plt.subplots(figsize=(4, 3.4), dpi=110)
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}\n{labels[i, j]}", ha="center", va="center",
                    fontsize=9, color="white" if matrix[i, j] > matrix.max() / 2 else TEXT)
    ax.set_xticks([0, 1], ["Pred: Stay", "Pred: Churn"], fontsize=9)
    ax.set_yticks([0, 1], ["Actual: Stayed", "Actual: Churned"], fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, fontweight="bold")
    fig.tight_layout()
    return fig


def importance_bar(importance, value_col="importance", title="Feature importance", top_n=12,
                   xlabel="Relative importance"):
    data = importance.head(top_n).iloc[::-1]
    fig, ax = _fig(7, 0.38 * len(data) + 1.3)
    ax.barh(data["feature"], data[value_col], color=STAYED_COLOR, height=0.6)
    _style(ax, title, xlabel, None)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def local_explanation_bar(explanation, title="Why this prediction?"):
    data = explanation.iloc[::-1]
    labels = [f"{f} = {v}" for f, v in zip(data["feature"], data["customer_value"])]
    colors = [INCREASE_COLOR if c > 0 else DECREASE_COLOR for c in data["contribution"]]
    fig, ax = _fig(7, 0.42 * len(data) + 1.4)
    ax.barh(labels, data["contribution"], color=colors, height=0.6)
    ax.axvline(0, color=MUTED, linewidth=1)
    _style(ax, title, "SHAP contribution (right = raises churn risk, left = lowers it)", None)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def risk_distribution(scored):
    counts = scored["risk_category"].value_counts().reindex(list(RISK_COLORS), fill_value=0)
    fig, ax = _fig(5.5, 3.4)
    bars = ax.bar(counts.index, counts.values, color=[RISK_COLORS[k] for k in counts.index],
                  width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.annotate(f"{value:,}", (bar.get_x() + bar.get_width() / 2, value), ha="center",
                    va="bottom", fontsize=9, color=TEXT, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, max(counts.max(), 1) * 1.18)
    _style(ax, "Customers by predicted risk category", None, "Customers")
    fig.tight_layout()
    return fig


def probability_histogram(scored, low, high):
    fig, ax = _fig(7, 3.6)
    ax.hist(scored["churn_probability"], bins=40, color=STAYED_COLOR, edgecolor="white")
    for x, name in [(low, "Medium"), (high, "High")]:
        ax.axvline(x, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"{name} risk from {x:.0%}", (x, ax.get_ylim()[1] * 0.92), fontsize=8,
                    color=MUTED, xytext=(3, 0), textcoords="offset points")
    _style(ax, "Distribution of predicted churn probability", "Churn probability", "Customers")
    fig.tight_layout()
    return fig


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def signup_trend(df, target="churn"):
    """Churn rate by signup year (date field analysis)."""
    data = df.dropna(subset=["signup_date"]).copy()
    data["signup_year"] = pd.to_datetime(data["signup_date"]).dt.year
    g = data.groupby("signup_year")[target].agg(["mean", "count"])
    g = g[g["count"] >= 30]
    fig, ax = _fig(7, 3.4)
    ax.plot(g.index, g["mean"] * 100, marker="o", color=CHURN_COLOR, linewidth=2)
    _style(ax, "Churn rate by signup year", "Signup year", "Churn rate (%)")
    fig.tight_layout()
    return fig
