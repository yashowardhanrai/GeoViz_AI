import streamlit as st
import plotly.express as px

from utils.data_loader import load_data


def show_conflict_analytics():

    df = load_data()

    latest = df.iloc[-1]

    st.title("⚔️ Conflict Analytics")

    # ============================================
    # KPI CARDS
    # ============================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Fatalities",
        int(latest["fatalities"])
    )

    col2.metric(
        "Event Count",
        int(latest["event_count"])
    )

    col3.metric(
        "Severity",
        round(latest["severity"], 2)
    )

    col4.metric(
        "Conflict Pressure",
        round(latest["conflict_pressure"], 2)
    )

    st.divider()

    # ============================================
    # Fatalities Trend
    # ============================================

    fig1 = px.line(

        df,

        x="date",

        y=[
            "fatalities",
            "fatalities_30d",
            "fatalities_90d"
        ],

        title="Fatalities Trend"

    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ============================================
    # Severity
    # ============================================

    fig2 = px.line(

        df,

        x="date",

        y=[
            "severity",
            "severity_7d",
            "severity_30d"
        ],

        title="Conflict Severity"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ============================================
    # Intensity
    # ============================================

    fig3 = px.line(

        df,

        x="date",

        y=[
            "intensity",
            "intensity_7d",
            "intensity_30d"
        ],

        title="Conflict Intensity"

    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # ============================================
    # Event Count
    # ============================================

    fig4 = px.area(

        df,

        x="date",

        y="event_count",

        title="Daily Event Count"

    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # ============================================
    # Conflict Pressure
    # ============================================

    fig5 = px.line(

        df,

        x="date",

        y="conflict_pressure",

        title="Conflict Pressure"

    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )