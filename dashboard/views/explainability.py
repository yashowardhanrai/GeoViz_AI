import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    load_shap_regression,
    load_shap_classifier,
    load_baselines,
)


def show_explainability():

    st.title("🧠 Model Explainability")

    st.caption(
        "SHAP values from trained XGBoost v6.3 models. "
        "Loaded from saved artefacts — no retraining on this page."
    )

    # ============================================
    # TABS
    # ============================================

    tab1, tab2, tab3 = st.tabs([
        "Regression SHAP (3-day)",
        "Classifier SHAP (7-day)",
        "Baseline comparison"
    ])

    # ============================================
    # TAB 1 — REGRESSION SHAP
    # ============================================

    with tab1:

        st.subheader("Feature importance — 3-day risk score forecast")
        st.caption(
            "Mean |SHAP| across 836 training rows. "
            "is_monday ranked #47/47 (0.027), confirming dayofweek "
            "captured oil market microstructure, not conflict seasonality."
        )

        try:
            shap_reg = load_shap_regression()
            top20 = shap_reg.head(20).sort_values("Importance")

            fig = px.bar(
                top20,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 20 SHAP features — regression model (3-day horizon)",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                height=550,
                coloraxis_showscale=False,
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                shap_reg.head(20).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        except FileNotFoundError:
            st.error(
                "shap_importance_v6.csv not found. "
                "Run models/xgboost_model.py first."
            )

    # ============================================
    # TAB 2 — CLASSIFIER SHAP
    # ============================================

    with tab2:

        st.subheader("Feature importance — 7-day regime change classifier")
        st.caption(
            "Mean |SHAP| from best OOF fold (fold 4, val AUC=0.873). "
            "events_30d_lag1 is the dominant predictor of regime transitions."
        )

        try:
            shap_clf = load_shap_classifier()
            top15 = shap_clf.head(15).sort_values("Importance")

            fig2 = px.bar(
                top15,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 SHAP features — regime change classifier (7-day)",
                color="Importance",
                color_continuous_scale="Greens",
            )
            fig2.update_layout(
                height=480,
                coloraxis_showscale=False,
                yaxis_title="",
            )
            st.plotly_chart(fig2, use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric("Classifier AUC", "0.815")
            col1.metric("95% CI", "[0.739, 0.878]")
            col2.metric("F1 @ threshold 0.15", "0.620")
            col2.metric("Brier (calibrated)", "0.060")

            st.dataframe(
                shap_clf.head(15).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        except FileNotFoundError:
            st.error(
                "shap_importance_regime_classifier_v3.csv not found. "
                "Run models/regime_classifier.py first."
            )

    # ============================================
    # TAB 3 — BASELINE COMPARISON
    # ============================================

    with tab3:

        st.subheader("Model vs baseline comparison")

        try:
            baselines = load_baselines()

            reg = baselines[baselines["task"] == "regression"].copy()
            clf = baselines[baselines["task"] == "classification"].copy()

            st.markdown("**Regression — test R² by horizon**")

            xgb_row = pd.DataFrame([{
                "task": "regression", "horizon": "1,3,7",
                "model": "XGBoost v6.3",
                "r2": "0.730 / 0.750 / 0.554",
                "mae": "1.144 / 1.151 / 1.478"
            }])

            reg_display = pd.concat([reg, xgb_row], ignore_index=True)
            st.dataframe(
                reg_display[["model", "horizon", "r2", "mae"]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "**Note:** ARIMA/AR(14) exploit autocorrelation directly. "
                "XGBoost adds conflict, oil, and news signal. "
                "The regime classifier (below) is where multi-feature capacity adds unique value."
            )

            st.divider()

            st.markdown("**Classification — 7-day regime change (OOF evaluation)**")
            st.caption(
                "Baselines evaluated on held-out test set (0 positives). "
                "XGBoost evaluated on walk-forward OOF (56 positives). "
                "OOF is the only valid comparison for rare-event classifiers."
            )

            clf_cols = [
                c for c in ["model", "auc", "ap", "f1", "precision", "recall", "brier"]
                if c in clf.columns
            ]
            st.dataframe(
                clf[clf_cols],
                use_container_width=True,
                hide_index=True,
            )

        except FileNotFoundError:
            st.warning(
                "paper_baselines.csv not found. "
                "Run models/baselines.py to generate it."
            )