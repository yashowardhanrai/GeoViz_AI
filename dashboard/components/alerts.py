import streamlit as st
import pandas as pd


def show_classifier_alert(oof_df: pd.DataFrame, threshold: float = 0.15):
    """
    Renders a prominent alert banner based on the latest OOF classifier prediction.
    Call this at the top of any page that should surface the regime-change alert.
    """
    if oof_df is None or len(oof_df) == 0:
        return

    latest = oof_df.iloc[-1]
    prob = float(latest.get("pred_cal", 0))
    fired = bool(latest.get("pred_label", 0))

    if fired:
        st.error(
            f"⚠️ **Regime change alert** — escalation probability **{prob:.3f}** "
            f"exceeds threshold ({threshold}). "
            "A transition to a higher risk regime is likely within 7 days.",
            icon="🚨",
        )
    else:
        st.success(
            f"No regime change alert — escalation probability {prob:.3f} "
            f"is below threshold ({threshold}).",
            icon="✅",
        )


def show_risk_level_banner(risk_score: float, regime: str):
    """Coloured banner for current risk level."""
    msgs = {
        "Stable":   ("✅ Stable environment — geopolitical risk is low.", "success"),
        "Elevated": ("🟡 Elevated risk — monitor for developments.", "warning"),
        "High":     ("🟠 High risk — stressed geopolitical conditions.", "warning"),
        "Crisis":   ("🔴 Crisis — extreme uncertainty. Immediate attention required.", "error"),
    }
    msg, kind = msgs.get(regime, ("Unknown regime.", "info"))
    getattr(st, kind)(f"{msg}  Risk score: **{round(risk_score, 2)}**")


def show_data_freshness(df: pd.DataFrame):
    """Small caption showing dataset date range."""
    if df is None or len(df) == 0:
        return
    latest_date = df["date"].max().strftime("%d %b %Y")
    earliest_date = df["date"].min().strftime("%d %b %Y")
    st.caption(
        f"Dataset: {earliest_date} → {latest_date} · "
        f"{len(df):,} rows · "
        "Ukraine · Russia · Israel · Palestine · Iran"
    )