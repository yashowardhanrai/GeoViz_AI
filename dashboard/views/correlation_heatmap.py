import streamlit as st
import plotly.express as px

from utils.data_loader import load_data


def show_correlation_heatmap():

    st.title("📊 Feature Correlation Heatmap")

    df = load_data()

    cols = [
        "risk_score", "Close", "daily_return", "volatility_30",
        "gold_close", "sp500_close", "dxy_close",
        "fatalities", "intensity", "article_count", "avg_tone",
        "market_stress", "conflict_pressure", "media_pressure"
    ]
    cols = [c for c in cols if c in df.columns]

    corr_matrix = df[cols].corr().round(2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation matrix — key features"
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strongest positive correlations with risk_score")

    corr_pairs = (
        corr_matrix[["risk_score"]]
        .drop("risk_score")
        .reset_index()
    )
    corr_pairs.columns = ["Feature", "Correlation"]
    corr_pairs = corr_pairs.sort_values("Correlation", ascending=False)

    st.dataframe(corr_pairs, use_container_width=True, hide_index=True)