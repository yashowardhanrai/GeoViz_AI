import streamlit as st
import plotly.express as px

from utils.data_loader import load_data


def show_overview():

    df = load_data()

    latest = df.iloc[-1]

    st.title("🌍 GeoVizAI")
    st.subheader(
        "AI-Powered Geopolitical Risk Intelligence Platform"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Risk Score",
        round(latest["risk_score"], 2)
    )

    col2.metric(
        "Risk Level",
        latest["risk_level"]
    )

    col3.metric(
        "Market Stress",
        round(latest["market_stress"], 2)
    )

    col4.metric(
        "Crisis Index",
        round(latest["crisis_index"], 2)
    )

    st.divider()

    fig = px.line(
        df,
        x="date",
        y=[
            "risk_score",
            "crisis_index"
        ],
        title="Risk Score vs Crisis Index"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    fig2 = px.line(
        df,
        x="date",
        y=[
            "global_pressure",
            "market_stress",
            "conflict_pressure",
            "media_pressure"
        ],
        title="Pressure Components"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )