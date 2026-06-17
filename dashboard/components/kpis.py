import streamlit as st
import pandas as pd


def show_risk_kpis(latest: pd.Series, df: pd.DataFrame):
    """
    Primary risk KPI row — risk score, level, 30d trend, crisis index.
    Pass the latest row and the full dataframe for delta calculations.
    """
    prev = df.iloc[-2] if len(df) > 1 else latest

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Risk score",
        round(latest["risk_score"], 2),
        delta=round(latest["risk_score"] - prev["risk_score"], 2),
    )

    col2.metric(
        "Risk level",
        latest.get("risk_level", latest.get("regime", "—")),
    )

    col3.metric(
        "30-day average",
        round(latest["risk_30d"], 2),
        delta=round(latest["risk_30d"] - prev["risk_30d"], 2),
    )

    col4.metric(
        "Crisis index",
        round(latest["crisis_index"], 2),
        delta=round(latest["crisis_index"] - prev["crisis_index"], 2),
    )


def show_market_kpis(latest: pd.Series, df: pd.DataFrame):
    """Oil price, daily return, volatility, market stress."""
    prev = df.iloc[-2] if len(df) > 1 else latest

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Oil price (Brent)",
        f"${round(latest['Close'], 2)}",
        delta=round(latest["Close"] - prev["Close"], 2),
    )

    col2.metric(
        "Daily return",
        f"{round(latest['daily_return'], 2)}%",
        delta=round(latest["daily_return"] - prev["daily_return"], 2),
    )

    col3.metric(
        "Volatility (30d)",
        round(latest["volatility_30"], 2),
        delta=round(latest["volatility_30"] - prev["volatility_30"], 2),
    )

    col4.metric(
        "Market stress",
        round(latest["market_stress"], 2),
        delta=round(latest["market_stress"] - prev["market_stress"], 2),
    )


def show_conflict_kpis(latest: pd.Series, df: pd.DataFrame):
    """Fatalities, event count, severity, conflict pressure."""
    prev = df.iloc[-2] if len(df) > 1 else latest

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Fatalities",
        int(latest["fatalities"]),
        delta=int(latest["fatalities"] - prev["fatalities"]),
    )

    col2.metric(
        "Event count",
        int(latest["event_count"]),
        delta=int(latest["event_count"] - prev["event_count"]),
    )

    col3.metric(
        "Severity",
        round(latest["severity"], 2),
        delta=round(latest["severity"] - prev["severity"], 2),
    )

    col4.metric(
        "Conflict pressure",
        round(latest["conflict_pressure"], 2),
        delta=round(latest["conflict_pressure"] - prev["conflict_pressure"], 2),
    )


def show_classifier_kpis():
    """Static model performance KPIs for the classifier."""
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC", "0.815")
    col2.metric("95% CI", "[0.739, 0.878]")
    col3.metric("F1 @ 0.15", "0.620")
    col4.metric("Precision", "0.614")
    col5.metric("Recall", "0.625")


def show_regression_kpis():
    """Static model performance KPIs for regression."""
    col1, col2, col3 = st.columns(3)
    col1.metric("1-day Test R²", "0.730", delta="-0.048 vs naive")
    col2.metric("3-day Test R²", "0.750", delta="-0.066 vs naive")
    col3.metric("7-day Test R²", "0.554", delta="-0.128 vs naive")