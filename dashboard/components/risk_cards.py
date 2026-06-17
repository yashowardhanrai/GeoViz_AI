import streamlit as st
import pandas as pd


REGIME_EMOJI = {
    "Stable":   "🟢",
    "Elevated": "🔵",
    "High":     "🟠",
    "Crisis":   "🔴",
}


def show_current_risk_card(latest: pd.Series):
    """
    Compact summary card: risk score, regime, classifier alert probability.
    Use in sidebars or at the top of overview pages.
    """
    score = round(latest["risk_score"], 2)
    regime = latest.get("regime", latest.get("risk_level", "Unknown"))
    emoji = REGIME_EMOJI.get(regime, "⚪")

    st.markdown(
        f"""
        <div style="
            background: var(--background-color, #f8f9fa);
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        ">
            <div style="font-size: 13px; color: #6c757d; margin-bottom: 4px;">
                Current risk score
            </div>
            <div style="font-size: 32px; font-weight: 600; line-height: 1.1;">
                {score}
            </div>
            <div style="font-size: 14px; margin-top: 6px;">
                {emoji} {regime}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_regime_summary_cards(df: pd.DataFrame):
    """
    Four cards showing how many days the dataset spent in each regime.
    """
    if "regime" not in df.columns and "risk_level" not in df.columns:
        return

    regime_col = "regime" if "regime" in df.columns else "risk_level"
    counts = df[regime_col].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    regimes = ["Stable", "Elevated", "High", "Crisis"]

    for col, regime in zip(cols, regimes):
        n = int(counts.get(regime, 0))
        pct = round(n / len(df) * 100, 1)
        emoji = REGIME_EMOJI.get(regime, "⚪")
        col.metric(
            f"{emoji} {regime}",
            f"{n} days",
            delta=f"{pct}% of history",
            delta_color="off",
        )


def show_model_performance_card():
    """
    Summary card of key model metrics for the paper.
    """
    st.markdown("#### Model performance summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Regression (XGBoost v6.3)**")
        st.markdown(
            "| Horizon | Test R² | Naive R² |\n"
            "|---------|---------|----------|\n"
            "| 1-day   | 0.730   | 0.778    |\n"
            "| 3-day   | 0.750   | 0.816    |\n"
            "| 7-day   | 0.554   | 0.682    |"
        )

    with col2:
        st.markdown("**Classifier (v3, OOF)**")
        st.markdown(
            "| Metric    | Value              |\n"
            "|-----------|--------------------|\n"
            "| AUC       | 0.815              |\n"
            "| 95% CI    | [0.739, 0.878]     |\n"
            "| F1        | 0.620 @ thresh 0.15|\n"
            "| Brier cal | 0.060              |"
        )