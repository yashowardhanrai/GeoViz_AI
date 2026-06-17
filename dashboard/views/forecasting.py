import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import load_data, load_predictions


def show_forecasting():

    st.title("📈 Risk Forecasting")

    df = load_data()

    # ============================================
    # XGBOOST — TEST SET PREDICTIONS
    # ============================================

    st.subheader("XGBoost v6.3 — test set predictions (3-day horizon)")
    st.caption(
        "Test R² = 0.750 · MAE = 1.151 · Naive R² = 0.816 · "
        "209 test rows (last 20% of timeline)"
    )

    try:
        preds = load_predictions()
        n = len(preds)
        test_dates = df["date"].iloc[-n:].values

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
            height=420,
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

        col1, col2, col3 = st.columns(3)
        col1.metric("1-day Test R²", "0.730", delta="-0.048 vs naive")
        col2.metric("3-day Test R²", "0.750", delta="-0.066 vs naive")
        col3.metric("7-day Test R²", "0.554", delta="-0.128 vs naive")

    except FileNotFoundError:
        st.warning(
            "predictions_v6_3d.csv not found. "
            "Run models/xgboost_model.py first."
        )

    st.divider()

    # ============================================
    # ALL HORIZONS — RISK SCORE HISTORY
    # ============================================

    st.subheader("Risk score history")

    roll_cols = [c for c in ["risk_score", "risk_7d", "risk_30d", "risk_90d"]
                 if c in df.columns]
    fig2 = px.line(
        df, x="date", y=roll_cols,
        title="Risk score with rolling averages",
        labels={"value": "Risk score", "variable": ""},
        color_discrete_map={
            "risk_score": "#E24B4A",
            "risk_7d":    "#378ADD",
            "risk_30d":   "#BA7517",
            "risk_90d":   "#639922",
        }
    )
    fig2.update_layout(
        height=360,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ============================================
    # BASELINE COMPARISON
    # ============================================

    st.subheader("Baseline comparison — all horizons")
    st.caption(
        "ARIMA and AR(14)-Ridge exploit autocorrelation directly. "
        "XGBoost adds conflict, oil, and news signal. "
        "None beat naive on regression — the classifier is where "
        "multi-feature XGBoost adds unique value."
    )

    baseline_data = {
        "Model":  ["Naive", "AR(14)-Ridge", "ARIMA (2,1,1)", "XGBoost v6.3"],
        "1-day R²": [0.778, 0.789, 0.844, 0.730],
        "3-day R²": [0.816, 0.829, 0.823, 0.750],
        "7-day R²": [0.682, 0.721, 0.709, 0.554],
        "1-day MAE": [1.002, 0.986, 0.834, 1.144],
        "3-day MAE": [0.971, 0.930, 0.897, 1.151],
        "7-day MAE": [1.172, 1.100, 1.067, 1.478],
    }
    st.dataframe(
        pd.DataFrame(baseline_data),
        use_container_width=True,
        hide_index=True,
    )