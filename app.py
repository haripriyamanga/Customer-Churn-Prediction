"""Customer Churn Prediction - ML Customer Retention Analytics dashboard.

Run with:
    streamlit run app.py

The dashboard only LOADS saved models (created by `python src/train_models.py`).
It never retrains silently; if models are missing it offers a clearly labelled
"Train models now" button.
"""

import subprocess
import sys

import pandas as pd
import streamlit as st

from src import plots
from src.config import (
    CATEGORICAL_VALUES,
    DATA_PATH,
    HIGH_RISK_MIN,
    LOW_RISK_MAX,
    NUMERIC_COLUMNS,
    PROJECT_ROOT,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMN,
)
from src.data_processing import clean_data, load_raw_data, validate_input_data
from src.evaluate_models import (
    METRIC_COLUMNS,
    comparison_table,
    confusion_matrix_frame,
    evaluate_predictions,
)
from src.explainability import (
    SHAP_AVAILABLE,
    explain_customer,
    model_feature_importance,
    shap_output_units,
)
from src.feature_engineering import FEATURE_DESCRIPTIONS, engineer_features
from src.insights import (
    SEGMENT_COLUMNS,
    churn_rate_by,
    generate_insights,
    retention_recommendations,
)
from src.predictions import (
    ModelNotTrainedError,
    load_all_models,
    load_evaluation,
    load_primary_model,
    predict_single,
    score_customers,
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1300px;}
      .risk-badge {display:inline-block; padding:0.35rem 0.9rem; border-radius:999px;
                   font-weight:600; color:#0b0b0b; border:1px solid rgba(0,0,0,0.15);}
      .small-note {color:#6b6a66; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_ICONS = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning the dataset...")
def get_data():
    raw = load_raw_data(DATA_PATH)
    df, report = clean_data(raw)
    return raw, df, report.as_rows()


@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    return load_primary_model(), load_all_models(), load_evaluation()


@st.cache_data(show_spinner="Scoring customers with the primary model...")
def get_scored(_pipeline, model_name, trained_at):
    _, df, _ = get_data()
    return score_customers(_pipeline, df.drop(columns=[TARGET_COLUMN]),
                           threshold=get_models()[0]["classification_threshold"]).assign(
        churn=df[TARGET_COLUMN].values
    )


def show_fig(fig):
    st.pyplot(fig, width="stretch")
    plots.plt.close(fig)


def md_escape(text):
    """Escape '$' so Streamlit markdown does not render dollar amounts as LaTeX."""
    return str(text).replace("$", "\\$")


def pct(x):
    return f"{x:.1%}"


# ---------------------------------------------------------------------------
# Data + models (with friendly empty states)
# ---------------------------------------------------------------------------
st.title("📉 Customer Churn Prediction")
st.caption("ML customer retention analytics · synthetic demo data · no real customer information")

if not DATA_PATH.exists():
    st.error(
        f"The dataset was not found at `{DATA_PATH.relative_to(PROJECT_ROOT)}`. "
        "Generate it with `python src/generate_data.py`."
    )
    st.stop()

raw_df, df, cleaning_rows = get_data()

try:
    bundle, all_models, evaluation = get_models()
except ModelNotTrainedError:
    st.warning(
        "No trained models were found. The dashboard needs models created by the training "
        "script (`python src/train_models.py`)."
    )
    st.info(
        "You can train them now from here. This runs the full training pipeline "
        "(4 models, cross-validation and tuning) and usually takes 1-3 minutes."
    )
    if st.button("🚀 Train models now", type="primary"):
        with st.spinner("Training models... this can take a few minutes."):
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "src" / "train_models.py")],
                cwd=PROJECT_ROOT, capture_output=True, text=True,
            )
        if result.returncode == 0:
            st.success("Training finished. Reloading the dashboard...")
            get_models.clear()
            st.rerun()
        else:
            st.error("Training failed. See the log below.")
            st.code(result.stdout[-3000:] + "\n" + result.stderr[-3000:])
    st.stop()
except Exception as exc:  # e.g. incompatible library versions
    st.error(
        "The saved model files could not be loaded. This usually means they were created with "
        f"different library versions. Please retrain with `python src/train_models.py`.\n\n"
        f"Details: {exc}"
    )
    st.stop()

primary_name = bundle["model_name"]
primary = bundle["pipeline"]
scored = get_scored(primary, primary_name, bundle["trained_at"])
test_ids = set(evaluation["test_customer_ids"])
results = evaluation["models"]
table = comparison_table(results)
thresholds = {name: res["threshold"] for name, res in results.items()}
background = df.drop(columns=[TARGET_COLUMN]).sample(min(300, len(df)), random_state=42)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Predicts which customers are likely to churn, explains why, and suggests "
        "retention actions."
    )
    st.markdown(f"**Primary model:** {primary_name}")
    st.markdown(f"**Trained:** {bundle['trained_at']} (UTC)")
    st.markdown(f"**Customers:** {len(df):,}")
    st.divider()
    st.subheader("Risk categories")
    st.markdown(
        f"{RISK_ICONS['Low Risk']} **Low Risk**: probability < {LOW_RISK_MAX:.0%}  \n"
        f"{RISK_ICONS['Medium Risk']} **Medium Risk**: {LOW_RISK_MAX:.0%} – {HIGH_RISK_MIN:.0%}  \n"
        f"{RISK_ICONS['High Risk']} **High Risk**: probability ≥ {HIGH_RISK_MIN:.0%}"
    )
    st.caption(
        "Probabilities are model estimates, not certainties. The data is synthetic and "
        "for demonstration only."
    )

tabs = st.tabs([
    "📊 Overview",
    "🔍 Data Exploration",
    "🏆 Model Comparison",
    "🧮 Customer Prediction",
    "🧠 Explainability",
    "🚨 High-Risk Customers",
    "💡 Business Insights",
    "📤 Batch Scoring (CSV)",
])

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Overview")
    n_high = int((scored["risk_category"] == "High Risk").sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Churned customers", f"{int(df[TARGET_COLUMN].sum()):,}")
    c3.metric("Churn rate", pct(df[TARGET_COLUMN].mean()))
    c4.metric("Avg. churn probability", pct(scored["churn_probability"].mean()))
    c5.metric("High-risk customers", f"{n_high:,}")
    st.caption(
        f"Probabilities come from the primary model ({primary_name}). "
        f"{evaluation['n_train']:,} of these customers were used for training, so their scores "
        "are in-sample; test-set performance is reported in the Model Comparison tab."
    )

    left, right = st.columns(2)
    with left:
        show_fig(plots.risk_distribution(scored))
    with right:
        show_fig(plots.probability_histogram(scored, LOW_RISK_MAX, HIGH_RISK_MIN))

    st.markdown("#### Machine learning workflow")
    st.markdown(
        "Raw data → **cleaning** → **EDA** → **feature engineering** → stratified "
        "**train/test split** → **4 models** (Logistic Regression, Decision Tree, Random Forest, "
        "XGBoost) → **cross-validation & tuning** → **evaluation** → **explainability** "
        "→ **predictions & recommendations**"
    )
    with st.expander("🧹 Data cleaning report"):
        st.dataframe(pd.DataFrame(cleaning_rows, columns=["Step", "Count"]),
                     hide_index=True, width="stretch")
        st.caption(
            "Invalid values are set to missing (not deleted) and imputed inside the model "
            "pipeline using statistics from the training data only. Only rows with an unknown "
            "churn label and exact duplicates are removed."
        )

# ---------------------------------------------------------------------------
# 2. Data exploration
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Exploratory data analysis")
    st.info(
        "These charts show associations in the data. A higher churn rate in a group does not "
        "prove that the attribute causes churn.", icon="ℹ️"
    )
    left, right = st.columns([1, 1.4])
    with left:
        show_fig(plots.churn_distribution(df))
    with right:
        seg_label = st.selectbox(
            "Churn rate by segment", list(SEGMENT_COLUMNS.values()), index=0
        )
        seg_col = {v: k for k, v in SEGMENT_COLUMNS.items()}[seg_label]
        rates = churn_rate_by(df, seg_col)
        show_fig(plots.churn_rate_bar(rates, seg_col, f"Churn rate by {seg_label.lower()}",
                                      overall=df[TARGET_COLUMN].mean()))

    left, right = st.columns(2)
    with left:
        show_fig(plots.histogram_by_churn(df, "tenure_months", title="Tenure by churn status",
                                          xlabel="Tenure (months)"))
    with right:
        show_fig(plots.histogram_by_churn(df, "monthly_charges",
                                          title="Monthly charges by churn status",
                                          xlabel="Monthly charges ($)"))
    left, right = st.columns(2)
    with left:
        eng = engineer_features(df)
        corr_cols = NUMERIC_COLUMNS + ["service_count", "avg_monthly_spend", TARGET_COLUMN]
        show_fig(plots.correlation_heatmap(eng, corr_cols))
    with right:
        show_fig(plots.signup_trend(df))
        st.caption("Recent sign-ups have shorter tenure, so this trend largely mirrors the "
                   "tenure effect.")
    with st.expander("Preview the cleaned data"):
        st.dataframe(df.head(200), width="stretch")
    with st.expander("Preview the raw data (before cleaning)"):
        st.dataframe(raw_df.head(200), width="stretch")

# ---------------------------------------------------------------------------
# 3. Model comparison
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Model comparison")
    st.markdown(
        f"All metrics are computed on the held-out **test set** ({evaluation['n_test']:,} "
        f"customers, {pct(evaluation['test_churn_rate'])} churn). Each model labels a customer "
        "as a churner when its probability reaches the model's **Threshold**, chosen to "
        "maximise F1 on cross-validated training predictions (never on the test set). "
        f"Cross-validation (CV) scores use {evaluation['cv_folds']} folds of the training set."
    )
    fmt = {c: "{:.3f}" for c in table.columns}
    fmt["Threshold"] = "{:.2f}"
    st.dataframe(
        table.style.format(fmt).highlight_max(axis=0, color="#cde2fb",
                                              subset=[c for c in table.columns if c not in
                                                      ("Threshold", "CV ROC-AUC (std)")]),
        width="stretch",
    )
    st.success(f"**Primary model: {primary_name}.** {evaluation['selection_reason']}")

    left, right = st.columns(2)
    with left:
        show_fig(plots.model_comparison_bars(table))
    with right:
        show_fig(plots.roc_curves(results))

    st.markdown("#### Confusion matrices")
    cols = st.columns(4)
    for col, (name, res) in zip(cols, results.items()):
        with col:
            show_fig(plots.confusion_matrix_plot(res["test"]["confusion_matrix"], name))

    with st.expander("❓ False positives vs. false negatives - why accuracy is not enough"):
        cm = results[primary_name]["test"]["confusion_matrix"]
        st.markdown(
            f"""
- **False positive** (predicted churn, customer actually stayed): we spend retention budget
  (a discount, a call) on someone who wasn't leaving. The {primary_name} made **{cm['fp']}**
  of these on the test set.
- **False negative** (predicted stay, customer actually churned): we miss a customer we could
  have tried to keep - usually the more expensive mistake. The {primary_name} made
  **{cm['fn']}** of these.
- **Precision** = of the customers flagged as churners, how many really churned
  (few wasted offers).
- **Recall** = of the customers who really churned, how many we flagged (few missed churners).
- **F1** balances precision and recall. **ROC-AUC** measures ranking quality across all
  thresholds (0.5 = random, 1.0 = perfect).

With {pct(evaluation['test_churn_rate'])} churners in the test set, a model that always predicts
"stay" would be {pct(1 - evaluation['test_churn_rate'])} accurate and catch **zero** churners -
that is why accuracy alone is misleading here.
"""
        )

    with st.expander("🔧 Cross-validation, tuning and overfitting check"):
        rows = []
        for name, res in results.items():
            rows.append({
                "Model": name,
                "Baseline CV ROC-AUC": res["baseline_cv"]["roc_auc_mean"],
                "Final CV ROC-AUC": res["cv"]["roc_auc_mean"],
                "Final CV F1": res["cv"]["f1_mean"],
                "Tuned?": "Yes" if res["tuned"] else "No",
                "Train ROC-AUC": res["train_roc_auc"],
                "Test ROC-AUC": res["test"]["ROC-AUC"],
                "Best parameters": ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in res["best_params"].items()) or "defaults",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(
            {c: "{:.3f}" for c in ["Baseline CV ROC-AUC", "Final CV ROC-AUC", "Final CV F1",
                                   "Train ROC-AUC", "Test ROC-AUC"]}),
            width="stretch")
        st.caption(
            "A large gap between train and test ROC-AUC indicates overfitting (the model memorised "
            "training data). Hyperparameters were tuned with randomized search on training folds "
            f"only for: {', '.join(evaluation['tuned_models'])}."
        )

# ---------------------------------------------------------------------------
# 4. Customer prediction
# ---------------------------------------------------------------------------
def risk_badge(risk):
    color = plots.RISK_COLORS.get(risk, "#dddddd")
    return (f"<span class='risk-badge' style='background:{color}33;border-color:{color}'>"
            f"{RISK_ICONS.get(risk, '')} {risk}</span>")


def render_recommendations(recs):
    for rec in recs:
        tag = " · ✅ *supported by this customer's SHAP explanation*" if rec["supported_by_model"] else ""
        st.markdown(f"- **{rec['action']}**  \n  <span class='small-note'>{md_escape(rec['reason'])}</span>{tag}",
                    unsafe_allow_html=True)
    st.caption("These are suggested actions based on simple rules, not guaranteed solutions. "
               "Test them (e.g. with an A/B experiment) before rolling them out.")


with tabs[3]:
    st.subheader("Predict churn for a customer")
    st.write("Enter customer details below. Nothing is stored - the prediction happens in memory.")
    model_choice = st.selectbox(
        "Model", list(all_models), index=list(all_models).index(primary_name),
        help="The primary model is selected by default. Try others to compare.",
    )
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Customer profile**")
            gender = st.selectbox("Gender", CATEGORICAL_VALUES["gender"])
            age = st.number_input("Age", 18, 100, 35)
            partner = st.selectbox("Has partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            region = st.selectbox("Region", CATEGORICAL_VALUES["region"])
            satisfaction = st.slider("Satisfaction score (1 = very unhappy, 5 = very happy)", 1, 5, 3)
        with c2:
            st.markdown("**Account & billing**")
            tenure = st.number_input("Tenure (months)", 0, 120, 5)
            contract = st.selectbox("Contract type", CATEGORICAL_VALUES["contract_type"])
            payment = st.selectbox("Payment method", CATEGORICAL_VALUES["payment_method"], index=2)
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            monthly = st.number_input("Monthly charges ($)", 0.0, 500.0, 85.0, step=1.0)
            total = st.number_input(
                "Total charges ($)", 0.0, 60000.0, 425.0, step=10.0,
                help="Total billed so far. Roughly monthly charges × tenure.",
            )
            late = st.number_input("Late payments (last 12 months)", 0, 24, 1)
        with c3:
            st.markdown("**Services & support**")
            internet = st.selectbox("Internet service", CATEGORICAL_VALUES["internet_service"], index=1)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            no_net = internet == "No"
            services = st.multiselect(
                "Add-on services",
                ["online_security", "online_backup", "device_protection", "tech_support",
                 "streaming_tv", "streaming_movies"],
                default=["streaming_tv"],
                format_func=lambda s: s.replace("_", " ").capitalize(),
                help="Add-ons require internet service; they are ignored if Internet service = No.",
            )
            tickets = st.number_input("Support tickets (last 12 months)", 0, 50, 3)
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        customer = {
            "gender": gender, "age": age, "partner": partner, "dependents": dependents,
            "region": region, "tenure_months": tenure, "contract_type": contract,
            "payment_method": payment, "paperless_billing": paperless,
            "internet_service": internet, "phone_service": phone,
            "monthly_charges": monthly, "total_charges": total, "support_tickets": tickets,
            "satisfaction_score": satisfaction, "late_payments": late,
        }
        for s in ["online_security", "online_backup", "device_protection", "tech_support",
                  "streaming_tv", "streaming_movies"]:
            customer[s] = "Yes" if (s in services and not no_net) else "No"
        if tenure > 0 and total < monthly * 0.5:
            st.warning("Total charges look low for this tenure - please double-check the values.")
        st.session_state["prediction"] = (model_choice, customer)

    if "prediction" in st.session_state:
        model_name, customer = st.session_state["prediction"]
        pipe = all_models[model_name]
        with st.spinner("Scoring customer..."):
            out = predict_single(pipe, customer, threshold=thresholds[model_name])
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", out["prediction"])
        r2.metric("Churn probability", pct(out["probability"]))
        with r3:
            st.markdown("**Risk category**")
            st.markdown(risk_badge(out["risk_category"]), unsafe_allow_html=True)
        st.progress(min(max(out["probability"], 0.0), 1.0))
        st.caption(
            f"Model: {model_name} (flags churn at ≥ {thresholds[model_name]:.0%}). A probability of {pct(out['probability'])} means the model "
            "estimates this customer's profile is similar to customers who churned at about that "
            "rate - it is an estimate, not a certainty."
        )

        drivers = set()
        left, right = st.columns([1.3, 1])
        with left:
            st.markdown("#### Why this prediction?")
            if SHAP_AVAILABLE:
                try:
                    expl, _ = explain_customer(pipe, out["cleaned_input"], background)
                    show_fig(plots.local_explanation_bar(expl))
                    drivers = set(expl.loc[expl["contribution"] > 0, "feature"])
                    st.caption(
                        f"SHAP values in {shap_output_units(pipe)} units. Red bars pushed this "
                        "prediction towards churn, blue bars away from it. These show what "
                        "influenced the model, not proven causes of churn."
                    )
                except Exception as exc:
                    st.warning(f"A SHAP explanation could not be computed for this model: {exc}")
            else:
                st.info("Install `shap` to see a per-customer explanation.")
        with right:
            st.markdown("#### Suggested retention actions")
            recs = retention_recommendations(
                out["cleaned_input"].iloc[0].to_dict(), out["probability"], out["risk_category"],
                bundle["reference_values"], drivers,
            )
            render_recommendations(recs)

# ---------------------------------------------------------------------------
# 5. Explainability
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Explainable AI")
    st.info(
        "Feature importance and SHAP show which inputs most influence the model's predictions. "
        "They describe the model, not the real world - an influential feature is associated with "
        "churn in this data but is not proven to cause it.", icon="ℹ️"
    )
    left, right = st.columns(2)
    with left:
        exp_model = st.selectbox("Model-native feature importance", list(all_models),
                                 index=list(all_models).index(primary_name))
        imp, method = model_feature_importance(all_models[exp_model])
        show_fig(plots.importance_bar(imp, title=f"{exp_model}: {method.lower()}"))
        st.caption("One-hot encoded columns are summed back to their original feature.")
    with right:
        st.markdown(f"**Global explanations for the primary model ({primary_name})**")
        view = st.radio("Method", ["SHAP (mean |SHAP|)", "Permutation importance"], horizontal=True)
        if view.startswith("SHAP") and evaluation["shap_importance"]:
            show_fig(plots.importance_bar(pd.DataFrame(evaluation["shap_importance"]),
                                          "mean_abs_shap", "Mean |SHAP value| (test sample)",
                                          xlabel="Mean |SHAP value|"))
        elif view.startswith("SHAP"):
            st.info("SHAP importance was not computed (is `shap` installed?).")
        else:
            show_fig(plots.importance_bar(pd.DataFrame(evaluation["permutation_importance"]),
                                          title="Permutation importance (test set)",
                                          xlabel="Drop in ROC-AUC when shuffled"))
            st.caption("How much test ROC-AUC drops when a feature's values are shuffled. "
                       "Model-agnostic and computed on unseen data.")

    st.markdown("#### Why is this customer considered high risk?")
    test_scored = scored[scored["customer_id"].isin(test_ids)].sort_values(
        "churn_probability", ascending=False)
    options = test_scored["customer_id"].tolist()
    chosen = st.selectbox(
        "Choose a test-set customer (sorted from highest to lowest predicted risk)", options,
        format_func=lambda cid: (
            f"{cid} - {pct(float(test_scored.loc[test_scored.customer_id == cid, 'churn_probability'].iloc[0]))}"
            f" ({test_scored.loc[test_scored.customer_id == cid, 'risk_category'].iloc[0]})"
        ),
    )
    if chosen and SHAP_AVAILABLE:
        row = df[df["customer_id"] == chosen].drop(columns=[TARGET_COLUMN])
        srow = scored[scored["customer_id"] == chosen].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn probability", pct(srow["churn_probability"]))
        c2.metric("Risk category", f"{RISK_ICONS[srow['risk_category']]} {srow['risk_category']}")
        c3.metric("Actual outcome", "Churned" if srow["churn"] == 1 else "Stayed")
        with st.spinner("Computing SHAP explanation..."):
            expl, _ = explain_customer(primary, row, background)
        left, right = st.columns([1.3, 1])
        with left:
            show_fig(plots.local_explanation_bar(expl, f"Top factors for {chosen}"))
        with right:
            up = expl[expl["contribution"] > 0]
            if len(up):
                st.markdown("**Factors pushing the prediction towards churn** (from SHAP):")
                for _, r in up.iterrows():
                    st.markdown(f"- `{r['feature']}` = **{r['customer_value']}**")
            else:
                st.markdown("No factor pushes this customer's prediction towards churn.")
            st.caption(f"SHAP values in {shap_output_units(primary)} units.")
    with st.expander("Engineered feature definitions"):
        st.table(pd.DataFrame(FEATURE_DESCRIPTIONS.items(), columns=["Feature", "Definition"]))

# ---------------------------------------------------------------------------
# 6. High-risk customers
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("High-risk customers")
    st.write(f"Customers ranked by the primary model's ({primary_name}) churn probability.")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        risks = st.multiselect("Risk category", ["High Risk", "Medium Risk", "Low Risk"],
                               default=["High Risk"])
    with f2:
        contracts = st.multiselect("Contract type", CATEGORICAL_VALUES["contract_type"])
    with f3:
        min_prob = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)
    with f4:
        test_only = st.checkbox("Test-set customers only", value=False,
                                help="Test-set customers were not used for training, so their "
                                     "scores are an honest out-of-sample estimate.")
        exclude_churned = st.checkbox("Hide customers who already churned", value=True)

    view_df = scored.copy()
    view_df["in_test_set"] = view_df["customer_id"].isin(test_ids)
    if risks:
        view_df = view_df[view_df["risk_category"].isin(risks)]
    if contracts:
        view_df = view_df[view_df["contract_type"].isin(contracts)]
    view_df = view_df[view_df["churn_probability"] >= min_prob]
    if test_only:
        view_df = view_df[view_df["in_test_set"]]
    if exclude_churned:
        view_df = view_df[view_df["churn"] == 0]

    sort_col = st.selectbox("Sort by", ["churn_probability", "monthly_charges", "tenure_months",
                                        "support_tickets", "satisfaction_score"])
    ascending = st.toggle("Ascending order", value=False)
    view_df = view_df.sort_values(sort_col, ascending=ascending)

    show_cols = ["customer_id", "churn_probability", "risk_category", "contract_type",
                 "tenure_months", "monthly_charges", "payment_method", "internet_service",
                 "support_tickets", "satisfaction_score", "in_test_set"]
    if view_df.empty:
        st.info("No customers match these filters. Try widening them.")
    else:
        st.caption(f"{len(view_df):,} customers match.")
        st.dataframe(
            view_df[show_cols],
            hide_index=True,
            width="stretch",
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="percent"),
                "monthly_charges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
            },
        )
        st.download_button("⬇️ Download this list (CSV)",
                           view_df[show_cols].to_csv(index=False).encode(),
                           "high_risk_customers.csv", "text/csv")

# ---------------------------------------------------------------------------
# 7. Business insights
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Business Insights")
    st.caption("Every figure below is calculated from the data and the model's predictions.")
    shap_imp = pd.DataFrame(evaluation["shap_importance"]) if evaluation["shap_importance"] else None
    for text in generate_insights(df, scored, shap_imp):
        st.markdown(f"- {md_escape(text)}")

    left, right = st.columns(2)
    with left:
        show_fig(plots.churn_rate_bar(churn_rate_by(df, "contract_type"), "contract_type",
                                      "Churn rate by contract type",
                                      overall=df[TARGET_COLUMN].mean()))
    with right:
        show_fig(plots.churn_rate_bar(churn_rate_by(df, "tenure_group"), "tenure_group",
                                      "Churn rate by tenure group",
                                      overall=df[TARGET_COLUMN].mean()))

    st.markdown("#### Predicted risk by segment")
    seg_label = st.selectbox("Segment", list(SEGMENT_COLUMNS.values()), key="risk_seg")
    seg_col = {v: k for k, v in SEGMENT_COLUMNS.items()}[seg_label]
    eng_scored = engineer_features(scored) if seg_col not in scored else scored
    pivot = (pd.crosstab(eng_scored[seg_col], eng_scored["risk_category"], normalize="index")
             .reindex(columns=["Low Risk", "Medium Risk", "High Risk"], fill_value=0))
    st.dataframe(pivot.style.format("{:.1%}").background_gradient(cmap="Oranges",
                                                                   subset=["High Risk"]),
                 width="stretch")

    st.markdown("#### Retention recommendations for current high-risk customers")
    active_high = scored[(scored["risk_category"] == "High Risk") & (scored["churn"] == 0)]
    if active_high.empty:
        st.info("There are no active high-risk customers.")
    else:
        counts = {}
        for _, row in active_high.iterrows():
            for rec in retention_recommendations(row.to_dict(), row["churn_probability"],
                                                 row["risk_category"], bundle["reference_values"]):
                counts[rec["action"]] = counts.get(rec["action"], 0) + 1
        rec_df = (pd.DataFrame(counts.items(), columns=["Suggested action", "Customers"])
                  .sort_values("Customers", ascending=False))
        st.write(f"{len(active_high):,} customers who have not churned are currently High Risk. "
                 "The rules below would apply to them:")
        st.dataframe(rec_df, hide_index=True, width="stretch")
        st.caption("Rule-based suggestions, not guaranteed outcomes. One customer can match "
                   "several rules.")

# ---------------------------------------------------------------------------
# 8. Batch scoring (CSV upload)
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Score your own customer file")
    st.markdown(
        "Upload a CSV with the same columns as the demo dataset. The file is **scored with the "
        "already-trained model - no retraining happens.** A `churn` column is optional; if present "
        "(Yes/No), the model's accuracy on your file is also shown."
    )
    template = df.drop(columns=[TARGET_COLUMN]).head(5).copy()
    template["signup_date"] = template["signup_date"].dt.strftime("%Y-%m-%d")
    st.download_button("⬇️ Download a template CSV", template.to_csv(index=False).encode(),
                       "churn_template.csv", "text/csv")
    with st.expander("Required columns and allowed values"):
        st.write("Numeric: " + ", ".join(f"`{c}`" for c in NUMERIC_COLUMNS))
        for col, vals in CATEGORICAL_VALUES.items():
            st.write(f"`{col}`: " + ", ".join(vals))

    upload_model = st.selectbox("Model to use", list(all_models),
                                index=list(all_models).index(primary_name), key="upload_model")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            up_raw = pd.read_csv(uploaded, dtype=str)
        except Exception as exc:
            st.error(f"The file could not be read as CSV: {exc}")
            up_raw = None
        if up_raw is not None:
            check = validate_input_data(up_raw)
            for err in check["errors"]:
                st.error(err)
            for warn in check["warnings"]:
                st.warning(warn)
            if not check["errors"]:
                up_clean, up_report = clean_data(up_raw, require_target=False)
                if up_clean.empty:
                    st.error("No rows are left after cleaning.")
                else:
                    with st.spinner("Scoring uploaded customers..."):
                        up_scored = score_customers(all_models[upload_model], up_clean,
                                                    threshold=thresholds[upload_model])
                    st.success(f"Scored {len(up_scored):,} customers with {upload_model}.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Customers scored", f"{len(up_scored):,}")
                    m2.metric("Avg. churn probability", pct(up_scored["churn_probability"].mean()))
                    m3.metric("High-risk customers",
                              f"{int((up_scored['risk_category'] == 'High Risk').sum()):,}")
                    show_fig(plots.risk_distribution(up_scored))
                    labelled = (TARGET_COLUMN in up_scored
                                and up_scored[TARGET_COLUMN].notna().sum() > 0)
                    if labelled:
                        known = up_scored[up_scored[TARGET_COLUMN].notna()]
                        if known[TARGET_COLUMN].nunique() == 2:
                            res = evaluate_predictions(known[TARGET_COLUMN].astype(int),
                                                       known["churn_probability"],
                                                       thresholds[upload_model])
                            st.markdown("**Performance on your labelled rows**")
                            st.dataframe(pd.DataFrame([{m: res[m] for m in METRIC_COLUMNS}])
                                         .style.format("{:.3f}"), hide_index=True)
                            st.dataframe(confusion_matrix_frame(res["confusion_matrix"]))
                    out_cols = [c for c in ["customer_id"] + REQUIRED_INPUT_COLUMNS
                                if c in up_scored] + ["churn_probability", "predicted_churn",
                                                      "risk_category"]
                    st.dataframe(up_scored[out_cols].sort_values("churn_probability",
                                                                 ascending=False),
                                 hide_index=True, width="stretch")
                    st.download_button("⬇️ Download scored file",
                                       up_scored[out_cols].to_csv(index=False).encode(),
                                       "scored_customers.csv", "text/csv")
                    with st.expander("Cleaning report for your file"):
                        st.dataframe(pd.DataFrame(up_report.as_rows(), columns=["Step", "Count"]),
                                     hide_index=True)

st.divider()
st.caption(
    "Built with scikit-learn, XGBoost, SHAP and Streamlit · Synthetic demo data · "
    "Predictions are probabilistic estimates and should support, not replace, human judgement."
)
