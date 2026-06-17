import os
import pandas as pd
import streamlit as st

# data_loader.py lives at: GeoVizAI/dashboard/utils/data_loader.py
# Project root is:         GeoVizAI/
# So we go up 3 levels:    utils -> dashboard -> GeoVizAI
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


@st.cache_data
def load_data():
    path = os.path.join(
        PROJECT_ROOT, "data", "processed", "geoviz_risk_dataset.csv"
    )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def load_shap_regression():
    path = os.path.join(PROJECT_ROOT, "models", "shap_importance_v6.csv")
    return pd.read_csv(path)


@st.cache_data
def load_shap_classifier():
    path = os.path.join(
        PROJECT_ROOT, "models", "shap_importance_regime_classifier_v3.csv"
    )
    return pd.read_csv(path)


@st.cache_data
def load_predictions():
    path = os.path.join(PROJECT_ROOT, "models", "predictions_v6_3d.csv")
    return pd.read_csv(path)


@st.cache_data
def load_oof_classifier():
    path = os.path.join(
        PROJECT_ROOT, "models", "oof_predictions_regime_classifier_7d_v3.csv"
    )
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_baselines():
    path = os.path.join(PROJECT_ROOT, "models", "paper_baselines.csv")
    return pd.read_csv(path)


@st.cache_data
def load_dm_tests():
    path = os.path.join(PROJECT_ROOT, "models", "dm_test_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_delong_tests():
    path = os.path.join(PROJECT_ROOT, "models", "delong_test_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_iran_oos_results():
    path = os.path.join(PROJECT_ROOT, "models", "oos_iran_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_iran_oos_predictions():
    path = os.path.join(PROJECT_ROOT, "models", "oos_iran_predictions.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return None