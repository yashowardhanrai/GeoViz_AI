import streamlit as st
import plotly.express as px

from utils.data_loader import load_data


def show_media_intelligence():

    st.title("📰 Media Intelligence")

    df = load_data()

    # ===================================================
    # ARTICLE COUNT
    # ===================================================

    st.subheader("Article count")

    fig = px.line(
        df,
        x="date",
        y="article_count",
        title="Daily news volume"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # MEDIA ATTENTION INDEX
    # ===================================================

    st.subheader("Media attention index")

    fig = px.line(
        df,
        x="date",
        y="article_count_norm",
        title="Normalised media attention"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # TONE ANALYSIS
    # ===================================================

    st.subheader("Average tone")

    fig = px.line(
        df,
        x="date",
        y="avg_tone",
        title="News sentiment (GDELT avg tone)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # NEWS CHANGE (was news_momentum — column does not exist)
    # news_change = daily diff of article_count, built in dataset_builder
    # ===================================================

    if "news_change" in df.columns:
        st.subheader("News volume change")
        fig = px.line(
            df,
            x="date",
            y="news_change",
            title="Daily change in news volume"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # TONE CHANGE (was tone_momentum — column does not exist)
    # tone_change = daily diff of avg_tone, built in dataset_builder
    # ===================================================

    if "tone_change" in df.columns:
        st.subheader("Tone change")
        fig = px.line(
            df,
            x="date",
            y="tone_change",
            title="Daily change in news tone"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # 30-DAY TONE
    # ===================================================

    if "tone_30d" in df.columns:
        st.subheader("30-day tone moving average")
        fig = px.line(
            df,
            x="date",
            y="tone_30d",
            title="30-day rolling average tone"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================================================
    # MEDIA PRESSURE
    # ===================================================

    st.subheader("Media pressure index")

    fig = px.line(
        df,
        x="date",
        y="media_pressure",
        title="Media pressure index"
    )
    st.plotly_chart(fig, use_container_width=True)