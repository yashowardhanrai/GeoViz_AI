import streamlit as st


def build_sidebar():

    st.sidebar.title("GeoVizAI")
    st.sidebar.caption(
        "AI-powered geopolitical risk intelligence · "
        "Ukraine · Russia · Israel · Palestine · Iran"
    )

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Risk Monitor",
            "Market Analytics",
            "Macro Markets",
            "Conflict Analytics",
            "Media Intelligence",
            "Correlation Heatmap",
            "Forecasting",
            "Explainability",
            "AI Insights",
            "Model Results",
        ]
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "XGBoost v6.3 · Regime Classifier v3  \n"
        "1,075 rows · Feb 2022 – Jun 2025  \n"
        "AUC 0.815 [0.739, 0.878]"
    )

    return page