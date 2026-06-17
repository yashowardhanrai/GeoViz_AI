import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data


def show_market_analytics():

    df = load_data()
    latest = df.iloc[-1]

    st.title("📈 Market Analytics")

    # ============================================
    # KPI CARDS
    # ============================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Oil Price (Brent)",
        f"${round(latest['Close'], 2)}"
    )

    col2.metric(
        "Daily Return",
        f"{round(latest['daily_return'], 2)}%"
    )

    col3.metric(
        "Volatility (30d)",
        round(latest["volatility_30"], 2)
    )

    col4.metric(
        "Market Stress",
        round(latest["market_stress"], 2)
    )

    st.divider()

    # ============================================
    # OIL PRICE
    # ============================================

    fig1 = px.line(
        df,
        x="date",
        y="Close",
        title="Brent Crude Oil Price (USD)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ============================================
    # MULTI-ASSET
    # ============================================

    asset_cols = [
        c for c in ["Close", "gold_close", "sp500_close", "dxy_close"]
        if c in df.columns
    ]

    if len(asset_cols) > 1:
        fig2 = px.line(
            df,
            x="date",
            y=asset_cols,
            title="Multi-Asset Price Trends"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ============================================
    # VOLATILITY
    # ============================================

    vol_cols = [
        c for c in
        ["volatility_30", "gold_volatility", "sp500_volatility", "dxy_volatility"]
        if c in df.columns
    ]

    fig3 = px.line(
        df,
        x="date",
        y=vol_cols,
        title="30-Day Volatility by Asset"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ============================================
    # RETURNS
    # ============================================

    ret_cols = [
        c for c in
        ["daily_return", "gold_return", "sp500_return", "dxy_return"]
        if c in df.columns
    ]

    fig4 = px.line(
        df,
        x="date",
        y=ret_cols,
        title="Daily Returns by Asset"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ============================================
    # MARKET STRESS
    # ============================================

    fig5 = px.area(
        df,
        x="date",
        y="market_stress",
        title="Market Stress Index"
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ============================================
    # OIL vs RISK SCORE
    # ============================================

    fig6 = px.scatter(
        df,
        x="Close",
        y="risk_score",
        color="regime",
        title="Oil Price vs Risk Score by Regime",
        labels={"Close": "Oil Price (USD)", "risk_score": "Risk Score"},
        color_discrete_map={
            "Stable": "#639922",
            "Elevated": "#378ADD",
            "High": "#BA7517",
            "Crisis": "#E24B4A"
        }
    )
    st.plotly_chart(fig6, use_container_width=True)