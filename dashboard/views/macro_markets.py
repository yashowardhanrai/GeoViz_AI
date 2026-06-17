import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data
from components.kpis import show_market_kpis


def show_macro_markets():

    df = load_data()
    latest = df.iloc[-1]

    st.title("🏦 Macro Markets")
    st.caption("Cross-asset market indicators and their relationship to geopolitical risk.")

    # ============================================
    # KPI ROW
    # ============================================

    show_market_kpis(latest, df)

    st.divider()

    # ============================================
    # GOLD vs OIL vs RISK
    # ============================================

    st.subheader("Gold vs oil vs risk score")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["Close"],
        name="Oil (Brent)", yaxis="y1",
        line=dict(color="#BA7517", width=2)
    ))

    if "gold_close" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["gold_close"],
            name="Gold", yaxis="y2",
            line=dict(color="#639922", width=2)
        ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["risk_score"],
        name="Risk score", yaxis="y3",
        line=dict(color="#E24B4A", width=1.5, dash="dot")
    ))

    fig.update_layout(
        height=400,
        hovermode="x unified",
        yaxis=dict(title=dict(text="Oil (USD)", font=dict(color="#BA7517"))),
        yaxis2=dict(title=dict(text="Gold (USD)", font=dict(color="#639922")),
                    overlaying="y", side="right"),
        yaxis3=dict(title=dict(text="Risk score", font=dict(color="#E24B4A")),
                    overlaying="y", side="right", anchor="free", position=0.95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # S&P 500 AND DXY
    # ============================================

    sp_cols = [c for c in ["sp500_close", "dxy_close"] if c in df.columns]
    if sp_cols:
        st.subheader("S&P 500 and DXY")
        fig2 = px.line(
            df, x="date", y=sp_cols,
            title="S&P 500 and US Dollar Index",
            color_discrete_map={
                "sp500_close": "#378ADD",
                "dxy_close": "#1D9E75",
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ============================================
    # 30-DAY RETURNS COMPARISON
    # ============================================

    ret_cols = [c for c in
                ["return_30d", "gold_return_30d", "sp500_return_30d", "dxy_return_30d"]
                if c in df.columns]

    if ret_cols:
        st.subheader("30-day returns by asset")
        fig3 = px.line(
            df, x="date", y=ret_cols,
            title="30-day rolling returns (%)"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ============================================
    # VOLATILITY COMPARISON
    # ============================================

    vol_cols = [c for c in
                ["volatility_30", "gold_volatility", "sp500_volatility", "dxy_volatility"]
                if c in df.columns]

    if vol_cols:
        st.subheader("Cross-asset volatility (30-day)")
        fig4 = px.line(
            df, x="date", y=vol_cols,
            title="30-day volatility by asset"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ============================================
    # MARKET STRESS vs RISK SCORE SCATTER
    # ============================================

    st.subheader("Market stress vs risk score")

    fig5 = px.scatter(
        df,
        x="market_stress",
        y="risk_score",
        color="regime" if "regime" in df.columns else None,
        color_discrete_map={
            "Stable":   "#639922",
            "Elevated": "#378ADD",
            "High":     "#BA7517",
            "Crisis":   "#E24B4A",
        },
        title="Market stress vs risk score (all dates)",
        labels={
            "market_stress": "Market stress",
            "risk_score": "Risk score",
        },
        opacity=0.6,
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ============================================
    # CORRELATION TABLE
    # ============================================

    st.subheader("Correlation with risk score")

    market_cols = [c for c in [
        "Close", "daily_return", "volatility_30", "return_30d",
        "gold_close", "gold_return", "gold_volatility",
        "sp500_close", "sp500_return", "sp500_volatility",
        "dxy_close", "dxy_return", "dxy_volatility",
        "market_stress",
    ] if c in df.columns]

    corr = (
        df[market_cols + ["risk_score"]]
        .corr()[["risk_score"]]
        .drop("risk_score")
        .rename(columns={"risk_score": "Correlation with risk score"})
        .sort_values("Correlation with risk score", ascending=False)
        .round(3)
    )

    st.dataframe(corr, use_container_width=True)