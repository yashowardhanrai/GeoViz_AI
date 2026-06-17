import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data


def show_risk_monitor():

    df = load_data()
    latest = df.iloc[-1]
    risk_score = float(latest["risk_score"])

    st.title("🚨 Risk Monitor")

    # ============================================
    # GAUGE — clean needle, no bar overlap
    # ============================================

    if risk_score < 25:
        zone_color = "#4CAF50"    # green  — Stable
    elif risk_score < 50:
        zone_color = "#2196F3"    # blue   — Elevated
    elif risk_score < 70:
        zone_color = "#FF9800"    # orange — High
    else:
        zone_color = "#F44336"    # red    — Crisis

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score, 2),
        title={"text": "Current Risk Score", "font": {"size": 16, "color": "white"}},
        number={"font": {"color": zone_color, "size": 52}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.3)",
                "tickfont": {"color": "rgba(255,255,255,0.6)"},
            },
            # Transparent bar so it doesn't overlay the coloured steps
            "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "#1B5E20"},   # dark green
                {"range": [25, 50], "color": "#0D47A1"},   # dark blue
                {"range": [50, 70], "color": "#E65100"},   # dark orange
                {"range": [70, 100], "color": "#B71C1C"},  # dark red
            ],
            "threshold": {
                "line": {"color": zone_color, "width": 6},
                "thickness": 0.85,
                "value": risk_score,
            },
        }
    ))

    gauge.update_layout(
        height=280,
        margin=dict(t=60, b=10, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )

    st.plotly_chart(gauge, use_container_width=True)

    # ============================================
    # KPI CARDS
    # ============================================

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", round(latest["risk_score"], 2))
    col2.metric("Risk Level",  latest["risk_level"])
    col3.metric("30D Risk",    round(latest["risk_30d"], 2))
    col4.metric("90D Risk",    round(latest["risk_90d"], 2))

    st.divider()

    # ============================================
    # RISK TREND
    # ============================================

    fig = px.line(
        df,
        x="date",
        y=["risk_score", "risk_7d", "risk_30d", "risk_90d"],
        title="Risk trend",
        color_discrete_map={
            "risk_score": "#F44336",
            "risk_7d":    "#2196F3",
            "risk_30d":   "#FF9800",
            "risk_90d":   "#4CAF50",
        },
        labels={"value": "Risk score", "variable": ""},
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.1)", range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ============================================
    # MOMENTUM
    # ============================================

    st.subheader("Risk momentum")

    col1, col2 = st.columns(2)

    fig2 = px.line(
        df,
        x="date",
        y=["risk_momentum", "risk_acceleration"],
        title="Momentum vs acceleration",
        color_discrete_map={
            "risk_momentum":     "#2196F3",
            "risk_acceleration": "#FF9800",
        },
        labels={"value": "", "variable": ""},
    )
    fig2.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
    )
    col1.plotly_chart(fig2, use_container_width=True)

    fig3 = px.area(
        df,
        x="date",
        y="risk_regime",
        title="Risk regime (numeric)",
        color_discrete_sequence=["#2196F3"],
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
    )
    col2.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ============================================
    # CRISIS INDICATORS
    # ============================================

    st.subheader("Crisis indicators")

    fig4 = px.line(
        df,
        x="date",
        y=["crisis_index", "global_pressure"],
        title="Crisis index vs global pressure",
        color_discrete_map={
            "crisis_index":    "#F44336",
            "global_pressure": "#FF9800",
        },
        labels={"value": "", "variable": ""},
    )
    fig4.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
    )
    st.plotly_chart(fig4, use_container_width=True)