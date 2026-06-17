import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# =====================================================
# RISK COLOUR HELPERS
# =====================================================

REGIME_COLORS = {
    "Stable":   "#639922",
    "Elevated": "#378ADD",
    "High":     "#BA7517",
    "Crisis":   "#E24B4A",
}

REGIME_BG = {
    "Stable":   "#EAF3DE",
    "Elevated": "#E6F1FB",
    "High":     "#FAEEDA",
    "Crisis":   "#FCEBEB",
}


def risk_color(score: float) -> str:
    if score >= 70:
        return "#E24B4A"
    if score >= 50:
        return "#BA7517"
    if score >= 25:
        return "#378ADD"
    return "#639922"


def regime_from_score(score: float) -> str:
    if score >= 70:
        return "Crisis"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Elevated"
    return "Stable"


# =====================================================
# GAUGE CHART
# =====================================================

def make_gauge(value: float, title: str = "Current Risk Score") -> go.Figure:
    color = risk_color(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 2),
        title={"text": title, "font": {"size": 16}},
        number={"font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  25], "color": "#EAF3DE"},
                {"range": [25, 50], "color": "#E6F1FB"},
                {"range": [50, 70], "color": "#FAEEDA"},
                {"range": [70, 100], "color": "#FCEBEB"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# =====================================================
# RISK TIMELINE
# =====================================================

def make_risk_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Regime background bands
    bands = [(0, 25, "#EAF3DE"), (25, 50, "#E6F1FB"),
             (50, 70, "#FAEEDA"), (70, 100, "#FCEBEB")]
    for lo, hi, color in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color,
                      opacity=0.3, line_width=0)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["risk_score"],
        mode="lines", name="Risk score",
        line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.07)"
    ))

    if "risk_30d" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["risk_30d"],
            mode="lines", name="30-day MA",
            line=dict(color="#BA7517", width=1.5, dash="dash")
        ))

    fig.update_layout(
        height=320,
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(range=[0, 100], gridcolor="rgba(128,128,128,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    return fig


# =====================================================
# SHAP BAR CHART
# =====================================================

def make_shap_bar(df: pd.DataFrame, title: str,
                  color: str = "#378ADD", top_n: int = 15) -> go.Figure:
    top = df.head(top_n).sort_values("Importance")
    fig = px.bar(
        top, x="Importance", y="Feature",
        orientation="h", title=title,
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        height=max(350, top_n * 30),
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
        yaxis_title="",
        showlegend=False,
    )
    return fig


# =====================================================
# REGRESSION COMPARISON BAR
# =====================================================

def make_regression_comparison() -> go.Figure:
    horizons = ["1-day", "3-day", "7-day"]
    models = {
        "XGBoost v6.3":  [0.730, 0.750, 0.554],
        "ARIMA (2,1,1)": [0.844, 0.823, 0.709],
        "AR(14)-Ridge":  [0.789, 0.829, 0.721],
        "Naive":         [0.778, 0.816, 0.682],
    }
    colors = ["#378ADD", "#1D9E75", "#888780", "#D3D1C7"]

    fig = go.Figure()
    for (name, vals), color in zip(models.items(), colors):
        fig.add_trace(go.Bar(
            name=name, x=horizons, y=vals,
            marker_color=color, text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0.4, 0.95],
                   title="Test R²",
                   gridcolor="rgba(128,128,128,0.1)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig