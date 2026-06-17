import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import (
    load_shap_regression,
    load_shap_classifier,
    load_predictions,
    load_oof_classifier,
    load_baselines,
)
from utils.dashboard import make_shap_bar, make_regression_comparison
from components.kpis import show_classifier_kpis, show_regression_kpis


def show_model_results():

    st.title("📊 Model Results")
    st.caption(
        "Full research results for the GeoVizAI paper. "
        "XGBoost v6.3 · Regime Classifier v3 · leakage audit · bootstrap CIs."
    )

    # ============================================
    # TABS
    # ============================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Regression",
        "Classifier",
        "Explainability",
        "Ablation study",
    ])

    # ============================================
    # TAB 1 — OVERVIEW
    # ============================================

    with tab1:

        st.subheader("Key metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Regression model")
            show_regression_kpis()
            st.caption(
                "None of the three horizons beat the naive persistence baseline. "
                "XGBoost v6.5 achieves test R²=0.896/0.875/0.810; naive persistence achieves 0.926/0.910/0.886 (DM p<0.001). "
                "because risk_score is strongly autocorrelated. XGBoost's multi-feature "
                "capacity adds value in the classifier, not the regression."
            )

        with col2:
            st.markdown("##### Regime change classifier")
            show_classifier_kpis()
            st.caption(
                "OOF AUC 0.8106 [0.7403, 0.8780] beats the trend heuristic (AUC ~0.48) "
                "by 0.33 points. Isotonic calibration reduced Brier score (0.0807 → 0.0633, 22% improvement). "
                "Validated on a single escalation cycle (2022–present)."
            )

        st.divider()

        st.subheader("Regression baseline comparison")
        fig = make_regression_comparison()
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("is_monday finding")
        st.info(
            "Replacing the raw `dayofweek` ordinal with a binary `is_monday` flag "
            "caused its SHAP rank to drop from **#1/47** (v6.2) to **#47/47** (v6.3, importance=0.027). "
            "This confirms dayofweek was capturing the oil market microstructure artifact "
            "(oil_ret_1 spans Fri→Mon = 3 calendar days on Mondays vs 1 day otherwise), "
            "not any generalizable conflict seasonality. "
            "Reported as a methodological finding in the paper."
        )

    # ============================================
    # TAB 2 — REGRESSION
    # ============================================

    with tab2:

        st.subheader("Test set predictions — 3-day horizon")

        try:
            preds = load_predictions()
            df_main = __import__(
                "utils.data_loader", fromlist=["load_data"]
            ).load_data()
            n = len(preds)
            test_dates = df_main["date"].iloc[-n:].values

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=test_dates, y=preds["Actual"],
                mode="lines", name="Actual",
                line=dict(color="#E24B4A", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=test_dates, y=preds["Predicted"],
                mode="lines", name="XGBoost predicted",
                line=dict(color="#378ADD", width=2, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=test_dates, y=preds["Naive_Baseline"],
                mode="lines", name="Naive baseline",
                line=dict(color="#888780", width=1, dash="dot")
            ))
            fig.update_layout(
                height=380,
                title="3-day ahead risk score — test set (last 20% of data)",
                xaxis_title="Date",
                yaxis_title="Risk score",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"predictions_v6_3d.csv not found. Run xgboost_model.py first. ({e})")

        st.divider()
        st.subheader("Baseline comparison table")

        try:
            bl = load_baselines()
            reg = bl[bl["task"] == "regression"][["model", "horizon", "r2", "mae"]]
            xgb_row = pd.DataFrame([{
                "model": "XGBoost v6.3", "horizon": "1",
                "r2": 0.896, "mae": 1.714,
            }, {
                "model": "XGBoost v6.3", "horizon": "3",
                "r2": 0.875, "mae": 1.890,
            }, {
                "model": "XGBoost v6.3", "horizon": "7",
                "r2": 0.810, "mae": 2.329,
            }])
            combined = pd.concat([reg, xgb_row], ignore_index=True)
            combined["horizon"] = combined["horizon"].astype(str) + "-day"
            combined["r2"] = combined["r2"].apply(
                lambda v: round(float(v), 4) if str(v).replace(".", "").replace("-", "").isnumeric() else v
            )
            st.dataframe(combined, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Run models/baselines.py to generate paper_baselines.csv.")

    # ============================================
    # TAB 3 — CLASSIFIER
    # ============================================

    with tab3:

        st.subheader("Classifier performance — OOF evaluation")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("AUC", "0.8106")
        col2.metric("95% CI", "[0.7403, 0.8780]")
        col3.metric("AP", "0.511")
        col4.metric("F1", "0.620")
        col5.metric("Brier raw", "0.0807")
        col6.metric("Brier cal", "0.0633")

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("##### Confusion matrix (OOF, calibrated)")
            cm_data = {
                "Predicted: No alert": [459, 21],
                "Predicted: Alert": [22, 35],
            }
            cm_df = pd.DataFrame(
                cm_data,
                index=["Actual: No change", "Actual: Change"]
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(
                "35 true alerts · 21 missed escalations · "
                "22 false alarms · 459 correct negatives"
            )

        with col_r:
            st.markdown("##### Calibration — Brier score")
            fig_b = go.Figure(go.Bar(
                x=["Raw probabilities", "After isotonic calibration"],
                y=[0.0807, 0.0633],
                marker_color=["#F7C1C1", "#9FE1CB"],
                text=["0.0807", "0.0633"],
                textposition="outside",
            ))
            fig_b.update_layout(
                height=240,
                yaxis=dict(range=[0, 0.14]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_b, use_container_width=True)

        st.divider()
        st.subheader("Alert history — OOF predictions")

        try:
            oof = load_oof_classifier()
            if "date" in oof.columns:
                fig_clf = px.scatter(
                    oof,
                    x="date",
                    y="pred_cal",
                    color=oof["pred_label"].map({1: "Alert fired", 0: "No alert"}),
                    color_discrete_map={
                        "Alert fired": "#E24B4A",
                        "No alert": "#B4B2A9",
                    },
                    title="Calibrated escalation probability — OOF timeline",
                    labels={
                        "pred_cal": "Escalation probability",
                        "color": "",
                    },
                    opacity=0.7,
                )
                fig_clf.add_hline(
                    y=0.15,
                    line_dash="dash",
                    line_color="#E24B4A",
                    annotation_text="Threshold 0.15",
                )
                fig_clf.update_layout(
                    height=340,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_clf, use_container_width=True)

        except Exception:
            st.info(
                "oof_predictions_regime_classifier_7d_v3.csv not found. "
                "Run models/regime_classifier.py first."
            )

    # ============================================
    # TAB 4 — EXPLAINABILITY
    # ============================================

    with tab4:

        col_reg, col_clf = st.columns(2)

        with col_reg:
            st.subheader("Regression SHAP (3-day)")
            try:
                shap_reg = load_shap_regression()
                fig_sr = make_shap_bar(
                    shap_reg, "Top 15 features — regression",
                    color="#378ADD", top_n=15
                )
                st.plotly_chart(fig_sr, use_container_width=True)
            except Exception:
                st.info("Run models/xgboost_model.py to generate shap_importance_v6.csv.")

        with col_clf:
            st.subheader("Classifier SHAP (7-day)")
            try:
                shap_clf = load_shap_classifier()
                fig_sc = make_shap_bar(
                    shap_clf, "Top 15 features — classifier",
                    color="#1D9E75", top_n=15
                )
                st.plotly_chart(fig_sc, use_container_width=True)
            except Exception:
                st.info(
                    "Run models/regime_classifier.py to generate "
                    "shap_importance_regime_classifier_v3.csv."
                )

    # ============================================
    # TAB 5 — ABLATION
    # ============================================

    with tab5:

        st.subheader("Leakage audit — ablation table")
        st.caption(
            "Each stage applies one fix to the pipeline. "
            "Fixed hyperparameters held constant across S1–S3 for fair comparison. "
            "S0, S4, S5 documented from prior runs (pipeline state unavailable for recompute)."
        )

        ablation = pd.DataFrame([
            {"Stage": "S0: Pre-audit (v5)",         "Fix": "Baseline — all leakage present",
             "1d R²": 0.621, "3d R²": 0.670, "7d R²": -0.089, "Note": "*"},
            {"Stage": "S1: +circular +leaky rank",   "Fix": "Add circular features back",
             "1d R²": -0.233, "3d R²": -0.108, "7d R²": -0.562, "Note": ""},
            {"Stage": "S2: −circular +leaky rank",   "Fix": "Remove oil_shock/oil_momentum",
             "1d R²": -0.233, "3d R²": -0.108, "7d R²": -0.562, "Note": ""},
            {"Stage": "S3: −circular, clean rank",   "Fix": "Fix expanding rank (risk_percentile)",
             "1d R²": -0.250, "3d R²": -0.101, "7d R²": -0.550, "Note": ""},
            {"Stage": "S4: +bfill fix +PCA fix",     "Fix": "Fix bfill→ffill, train-only PCA",
             "1d R²": 0.736, "3d R²": 0.755, "7d R²": 0.499,  "Note": "*"},
            {"Stage": "S5: per-horizon tuning",      "Fix": "Separate Optuna per horizon",
             "1d R²": 0.741, "3d R²": 0.740, "7d R²": 0.527,  "Note": "*"},
            {"Stage": "v6.3 final",                  "Fix": "STRING_AGG + is_monday",
             "1d R²": 0.896, "3d R²": 0.875, "7d R²": 0.810, "Note": ""},
        ])

        st.dataframe(
            ablation,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "* = documented from prior runs. "
            "Naive baseline: 1d=0.926, 3d=0.910, 7d=0.886 (date-based split)."
        )

        st.divider()
        st.subheader("7-day R² progression")

        fig_abl = go.Figure()
        fig_abl.add_trace(go.Scatter(
            x=ablation["Stage"],
            y=ablation["7d R²"],
            mode="lines+markers",
            line=dict(color="#378ADD", width=2),
            marker=dict(size=8, color="#378ADD"),
            name="7-day Test R²",
        ))
        fig_abl.add_hline(
            y=0.886,
            line_dash="dash",
            line_color="#888780",
            annotation_text="Naive baseline (0.886)",
        )
        fig_abl.add_hline(
            y=0,
            line_dash="dot",
            line_color="#E24B4A",
            annotation_text="R²=0 (no skill)",
        )
        fig_abl.update_layout(
            height=340,
            xaxis_tickangle=-30,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(
                title="Test R²",
                gridcolor="rgba(128,128,128,0.1)"
            ),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_abl, use_container_width=True)