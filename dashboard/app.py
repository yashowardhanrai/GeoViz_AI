import streamlit as st

from components.sidebar import build_sidebar

from views.overview import show_overview
from views.risk_monitor import show_risk_monitor
from views.market_analytics import show_market_analytics
from views.macro_markets import show_macro_markets
from views.conflict_analytics import show_conflict_analytics
from views.media_intelligence import show_media_intelligence
from views.correlation_heatmap import show_correlation_heatmap
from views.forecasting import show_forecasting
from views.explainability import show_explainability
from views.ai_insights import show_ai_insights
from views.model_results import show_model_results


st.set_page_config(
    page_title="GeoVizAI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = build_sidebar()

if page == "Overview":
    show_overview()

elif page == "Risk Monitor":
    show_risk_monitor()

elif page == "Market Analytics":
    show_market_analytics()

elif page == "Macro Markets":
    show_macro_markets()

elif page == "Conflict Analytics":
    show_conflict_analytics()

elif page == "Media Intelligence":
    show_media_intelligence()

elif page == "Correlation Heatmap":
    show_correlation_heatmap()

elif page == "Forecasting":
    show_forecasting()

elif page == "Explainability":
    show_explainability()

elif page == "AI Insights":
    show_ai_insights()

elif page == "Model Results":
    show_model_results()